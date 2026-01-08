def test_pyphi_plots_submodule_imports():
    import pyphi

    assert hasattr(pyphi, "plots")
    assert hasattr(pyphi.plots, "r2pv")
    assert hasattr(pyphi.plots, "loadings")
    assert hasattr(pyphi.plots, "score_scatter")


