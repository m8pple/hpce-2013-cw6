The bitecoin exercise was deliberately targetting multiple levels
of knowledge and performance. To get the best performance
required a combination of analysis, thought, design iteration,
observing what other people are doing, and parallelism. It was possible
to get good serial performance by, for example, doing a great deal
of analysis and thought, but as soon as you expose it on the
live exchange to test it, there is the risk that someone comes along
and adds parallelism.

So the broad levels of strategy I expected (and mostly
observed, though I've only started looking at actual code) are:

1 - Identifying basic parallelism within the reference code
===========================================================

The code contains a number of while and for loops, some of
which are data-parallel, some of which contain loop-carried
dependencies.

Obvious loop levels are:

1 - MakeBid : The outer loop over trials.
    This appears to have a loop carried dependency, one due
    to the minimum operation, the other due to the sequential
    calls to rand. However, we know that a minimum can be
    safely parallelised, and some thought (and inspection of
    HashReference), shows that it is not critical which values
    of rand are used.

2 - HashReference : The loop over indices.
    Here there is no dependency between loop iterations, except
    for the wide_xor at the end. xor is associative and commutative,
    so we could safely execute iterations in parallel and
    combine them using atomic xor ops.

3 - PoolHash : The loop over hashSteps.
    This has a clear loop carried depenency through x. There is no
    obvious way of parallelising the loop.

4 - wide_mul : The loops i and j over the limbs.
    There is a loop carried dependency through the i loop due
    to the carry/acc variables. Within the j loop there is
    also a dependency due to acc. So on the face of it, no data
    parallelism, but with some restructuring it is clear we
    could introduce some.

2 - Choosing the most appropriate parallelism
=============================================

So, having determined where the parallelism is, the task is
now to decide where best to exploit it. Exploiting wide_mul
may look attractive, but there is far too little work in there - the
cost of creating a task will outweigh the time to
execute ~64 multiply-accumulates. There is no point trying
to use CPU or GPU parallelism primitives to try to parallelise
a few hundred instructions. If we were in an FPGA, trying
to exploit SIMD, or were using VLIW, that would probably not
be the case. However, they all rely on low cost of work for
_static_ parallelism, where we know the schedule at compile time.

The loop over indices in HashReference is much more attractive,
as there is a sizeable chunk of work, involving probably 10000
or so instructions for a hashSteps of 16 or so. This is at the
edge of the efficiency boundary for tbb::parallel_for, but
is certainly at a task size where we can exploit GPU parallelism. 
The problem at this level is that we only aply PoolHash over
a few indices, probably not enough to fill up a work-group, and
not enough to amortise the cost of launching the kernel.

So the best starting point (which I think everyone will have
realised, possibly without thinking through every loop),
is to parallelise over MakeBid. Each point in the iteration
space involves a very large amount of work (~100000 instructions),
and there is an arbitrarily large iteration space to
allow us to launch large work groups.

The only drawback of parallelising here is that we need
to make sure we don't exceed the timelimit. One approach
is to try to guess the number of iterations, which works
fine, but requires a lot of overhead. Another approach
is to make sure that whichever thread sees the timer
expire first does the send:

    tbb::atomic<std::pair<bigint_t,vector<int> *> best;
    tbb::atomic<unsigned> haveSent=0;
    
    tbb::parallel_for(0u, n, [&](unsigned i){
        if(timeBudget<=0)
           return;
    
        auto indices=make_random_indices();
        auto proof=HashReference(pParams, indices);
        
        atomic_minimum(best, proof, indices);
        
        if(timeBudget<=0){
            // Guarantees only one thread sends
            if(haveSent++ == 0){
                td::pair<bigint_t,vector<int> *toGo=best.fetch_exchange(NULL);
                SendBid(toGo);
            }
        }
    });

Doing it in OpenCL is more tricky, but can be achieved
by using checking the timer on the CPU while work proceeds
asynchronously on the GPU.

At this point, you have a concrete plan for how to
parallelise, and have the option of trying it out to
check it works, or looking a bit further before you
start writing code.

3 - Identifying basic optimisations
===================================

While analysing to parallelise, you will hopefully have been looking
at the data dependencies shared by parallel iterations. So as
well as identifying dependencies which block parallelism, you
identify data which can be shared between parallel iterations,
saving computation.

The most obvious of these is the calculation of chainHash, as
even the simplest diagram will show that it is shared amongst
every parallel iteration we have identified. So pulling this
outside the MakeBid loop and doing it sequentially saves
calculating it thousands of times in parallel. This is an optimisation
that applies in both sequential and parallel code, but in
parallel code can help you with both compute time, and resource
usage - if you don't have to calculate chainHash on the GPU,
then you don't need all the memory traffic associated with it.

At this point it would be sensible to implement the parallel
version, to make sure you understand what is going on (and
again, I think most people did).

4 - Going deeper on the algorithm
=================================

At this point you should be able to get an approximately
linear speedup using either TBB or OpenCL, with OpenCL
probably a fair bit faster, but much higher coding overhead.
Hopefully you're seeing tens of thousands of hashes per round,
but for some reason your score isn't getting any better. And
certainly some other people are getting much better hashes
than you on the exchange.

So at this point you can:

1 - Stare at the code

2 - Look at what other people are doing

I think it took people a surprisingly long time to realise
how much value there was in option 2. I intentionally
put PonyExpress on there to show some possibilities, by
leaking strategies. (There were a few more four-legged
miners I had ready to bring in, but they weren't needed).

So looking at either of those two things, you hopefully
realise two things (this may have come later, in different
orders, etc.):

1 - maxIndices is definitely a max. You can return less
    than that (like PonyExpress) and the pool hash algorithm
    will still verify it as correct.

2 - The default distribution of indices is very limited,
    so DonkeyCart is very limited in the set of indices it
    returns. It must be retrying subsets of the same indices
    over and over, in slightly different combinations, and
    probably tries the same combination multiple times.

This hopefully leads to another couples of ideas:

3 - The default index generation algorithm only selects
    indices in the range 1..10*maxIndices, which is typically
    only about 320 indices. Every time we do a PoolHash we
    calculate the same indices over and over again. Why not
    calculate each index just once, and cache the proof?
    
4 - Going further, why are we limiting ourselves to so few
    indices? Lets assume we reduce maxIndices=4 for efficiency,
    and stick to the same index generating code. There are at most
    40 indices we would select from, and (remembering basic
    combinatorics), there are only binomial(40,4)=~2^16 possible
    combinations we could choose. So why restrict ourselves to
    such a small base set of indices?

I think a lot of people then got to the idea of a two
stage strategy:

1 - Generate proofs for a large number of indices.

2 - Try to find the best combination of the proofs from
    already calculated indices.

This restructures the algorithm a bit, but also makes
the parallelism cleaner and more explicit. The first
stage is eminently suitable for execution on a GPU
with OpenCL. You can burn through huge numbers of
indices, with your main limitation being setup and IO
time (depending on GPU). The second stage works well
in TBB or in OpenCL, and there is a good argument
for having it TBB: the sending of the solution is time
sensitive, so it is easier in software to grab the
best solution when the deadline expires.

This should get you well into the e+60 to e+70 range.

If you did this part well, and used parallelism correctly,
you're doing what is expected. So if you got to this
point, then well done.

5 - Down the rabbit hole
========================

Beyond this point you need to start thinking in much more
detail about combinatorics and what the hash function
is doing. For example, most people realised at some
level that you can use leading ones in the hash to
cancel each other out. Another strategy is to run your
PoolHash function over a huge number of indices, then
only use the smallest ones in the xor stage (as you
can guarantee that the leading zeros of all proofs
are the same).

Another strategy is just to watch what other people
are doing, and try to steal their strategy. I'm not
sure how many people tried this - the exchange deliberately
lets you do it, and I mentioned protecting IP a few times
in the lectures...

For example, Clockwork was doing consistently well, so it
it was worth seeing what they were doing. So you modify
the client to dump their solution, and I got something
like this on Friday afternoon:

[watcher], 1395421042.26, 2,          Clockwork : 8.7422e+33, 0.710351
     677223039 :   7013673e 6d9375a1 d67343a1 7bea4ddd 1321484b 9eb5b394 38087149 7ed6205e
     852613467 :   7013673e ca89f041 f4f697a2 b7fde5dc 60857f6c 6eed26a8 bfd11df6 0cbdee2f
    1446307660 :   06326ca9 057bff40 9b090200 c2064930 b802034a 1faeb649 9a4f6c1f 50725067
    1553243078 :   6df6691c 72177301 977d1863 e621031a ccfa8d14 e5bf1440 a40c46cb d0109ca2
    1621698088 :   06326ca9 627279e0 b98c5601 fe19e130 05663a6a efe6295e 221818cb de5a1e38
    1728633506 :   6df6691c cf0deda1 b6006c65 22349b1a 1a5ec435 b5f68755 2bd4f378 5df86a73
    2293591451 :   757636cc 159b865b 9a195124 813c9d74 c7733ef2 b1f51fb7 a058a66f 99f1a0d7
    2468981879 :   757636cc 729200fb b89ca525 bd503574 14d77613 822c92cc 2821531c 27d96ea8
    2687774582 :   0f07cbb0 35730da7 4e3b098c 04d50677 3e9a8f4c 9c7be0e3 253f8cdf f0a3a085
    2711703683 :   50cbecf1 272d8499 1f328335 2a93f993 5bb8a9e8 39fb5ed6 687e9fe6 e422d313
    2863165010 :   0f07cbb0 92698847 6cbe5d8d 40e89e76 8bfec66d 6cb353f7 ad08398c 7e8b6e56
    2887094111 :   50cbecf1 8423ff39 3db5d736 66a79192 a91ce109 0a32d1ea f0474c93 720aa0e4
    3128638075 :   267af9cf 6f4c0493 53126ac6 77e1dfa0 c8e996e6 22e87957 986ca711 1085c0bf
    3304028503 :   267af9cf cc427f33 7195bec7 b3f577a0 164dce06 f31fec6c 203553bd 9e6d8e90
    3492448923 :   105a80c0 32d63344 41ff688a cdaff87b 03e8e36a 60bb36e8 0e03df56 3c790baf
    3667839351 :   105a80c0 8fccade4 6082bc8c 09c3907a 514d1a8b 30f2a9fc 95cc8c02 ca60d980

                   00000000 00000000 00000000 00000000 0001af06 213f2242 0bb79975 af7299a4
                   00000000 00000000 00000000 00000000 0001af06 213f2242 0bb79975 af7299a4

We can immediately observe that they have managed to
cancel out the entire leading word, by choosing an
even number of indices, and making sure they contain
matches pairs.

A little thought even suggests how to do it: built a
hash-table indexed by the leading word, and as you
hash indices, detect any collisions and record them
as pairs. You can do this efficiently in either software,
or in the GPU using atomics (actually works very well).

You may wonder how many collisions you'd get, but if
you try it out, you'll find you actually get a lot (as
long as you can burn through indices). It is a variant
of the Birthday Paradox (see wikipedia), so if you
try ~2^16 indices, you should find ~2^16 colliding pairs.

Ok, first word down, how did they do the rest? There
are no cancelling words, but I notice that the number
of indices is 16, which is divisible by 4. Hrmm, I
wonder if they are looking for cancelling quads?

A few lines of matlab later, we find that there are
in fact some cancelling quads:

    >> uint32(find_zero_quads(x(:,3)))
    
    ans =
    
      1838380449  3398037569   896732583  2456389703
        92012352  1651669472   362514011  1922171131
      1914139393  3473796513   852898628  2412555748
       657294489  2216951609  1867252883  3426910003

so there are four unique quads which are cancelling out
the second word

Hrmm... 16 is also divisible by 8, what happens if I
try looking for octs in the 3rd word?:

    >> uint32(find_zero_octs(x(:,4)))

    ans =

      Columns 1 through 5

      3597878177  4109801378  2601058816  3112982017   523404085
      3597878177  4109801378  1312491916   523404085  1824415117
      2601058816  2541557859  3112982017  3053481061  2585350436
      2541557859  3053481061  2585350436  3097273637  1312491916

      Columns 6 through 8

      1035327286  1393715910  1905639111
      1035327286  1393715910  1905639111
      3097273637  1107257482  1619180684
      1824415117  1107257482  1619180684

Then obviously all 16 will cancel out for the 4th word.

So I don't know exactly how they are doing it, and in fact I
might be guessing a different strategy from them, but it
suggests one way of getting low hashes would be:

1 - Generate a huge pool of index pairs which cancel in
    the most significant word

2 - From within the pool of pairs, select pairs of indices
    which cancel in the 2nd word to give a pool of quads.

3 - From within the pool of quads, select pairs of indices
    which cancel in the third word to give a pool of quads.
    
and so on.

The combinatorics mean that it gets more and more difficult
to push down to lower words, but there is a clear advantage
to having more indices at the first level, as the number
of pairs will decrease as you go down.

Again: I have no idea how they did it (yet), but just
looking at their solution suggests an approach. I could
well be biased in the wrong direction, as this is simply
mapping their results onto something I'd already tried,
so maybe they did something very different which resulted
in similar looking results.

6 - Mathematical attacks
========================

There are also some fundamental weaknesses in the PoolHash
crypto function - as yet, I'm not sure whether anyone
found them, but I wouldn't be surprised. They aren't really
the point of the exercise anyway, more an amusing competition.
So I really don't expect people to have gone down this route,
and it isn't necessary to get decent marks.

So, drifting very far from the point of the exercise, PoolHash
is actually directly invertible (at least for the
fixed multiplier I chose). But I deliberately
tried to make it so you (and I) still can't get a zero
proof, as the server selects a number of components.

A much weaker system I considered would be to vary
the multiplier in PoolHashStep per round, and allow
the client to choose the 256 bit starting point x.
However, with knowledge of inversion, that would have
allowed a guaranteed zero with one index in constant time.

So the resulting system was an attempt to have
something that was hackable at every level: algorithmically,
with parallelism, and (to a certain extent) with maths.
