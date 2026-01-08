def test_pyphi_batch_submodule_imports():
    import pyphi

    assert hasattr(pyphi, "batch")
    assert hasattr(pyphi.batch, "mpca")
    assert hasattr(pyphi.batch, "mpls")
    assert hasattr(pyphi.batch, "monitor")
    assert hasattr(pyphi.batch, "phase_iv_align")


