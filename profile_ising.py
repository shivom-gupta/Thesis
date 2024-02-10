from cProfile import Profile
from ising import monte_carlo
import numpy as np
import pstats

profiler = Profile()

conf = np.random.choice([-1, 1], 1000)
profiler.runcall(monte_carlo, 1000, 100_000, 1, 1, 0, conf, None, None)

stats = pstats.Stats(profiler)
stats.strip_dirs()
stats.sort_stats(pstats.SortKey.TIME)
stats.dump_stats('profile_ising.prof')