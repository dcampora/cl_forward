General strategy of the algorithm

Triplets are chosen based on a fit and forwarded using a typical track following algo.

Ghosts are inherently out of the equation, as the algorithm considers all possible triplets and keeps the best. Upon following, if a hit is not found in the adjacent module, the track[let] is considered complete. Clones are removed based off a used-hit mechanism. A global array keeps track of used hits when forming tracks consist of 4 or more hits.

The algorithm consists in two stages: Track following, and seeding. In each step [iteration], the track following is performed first, hits are marked as used, and then the seeding is performed, requiring the first two hits in the triplet to be unused.
