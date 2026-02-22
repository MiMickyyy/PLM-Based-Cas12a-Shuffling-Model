from cas12a_shuffling_model.io.loaders import parse_domain_filename


def test_parse_domain_filename_basic():
    d = parse_domain_filename("As_01.dna")
    assert d.parent == "As"
    assert d.slot == 1


def test_parse_domain_filename_mb_normalizes_to_mb2():
    d = parse_domain_filename("Mb_11.dna")
    assert d.parent == "Mb2"
    assert d.slot == 11


def test_parse_domain_filename_rejects_unexpected():
    try:
        parse_domain_filename("BadName.txt")
    except ValueError:
        return
    assert False, "Expected ValueError"

