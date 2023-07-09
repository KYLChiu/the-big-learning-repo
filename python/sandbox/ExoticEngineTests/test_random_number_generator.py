from ExoticEngine.Statistics import RandomNumberGenerators as RNG


def test_RNG_set_dimension():
    old_dim = 19
    RNGenerator = RNG.TestRandom(dimension=old_dim)
    gaussians = RNGenerator.get_gaussian()
    assert len(gaussians) == old_dim
    new_dim = 132
    RNGenerator.reset_dimension(new_dim)
    gaussians = RNGenerator.get_gaussian()
    assert len(gaussians) == new_dim
