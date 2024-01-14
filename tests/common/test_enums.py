from bocoel import StrEnum


def test_str_enum_lookup():
    class Enum(StrEnum):
        A = "a"
        B = "b"
        C = "c"

    assert Enum.lookup(Enum.A) is Enum.A
    assert Enum.lookup("A") is Enum.A
    assert Enum.lookup("a") is Enum.A

    assert Enum.lookup(Enum.B) is Enum.B
    assert Enum.lookup("B") is Enum.B
    assert Enum.lookup("b") is Enum.B

    assert Enum.lookup(Enum.C) is Enum.C
    assert Enum.lookup("C") is Enum.C
    assert Enum.lookup("c") is Enum.C
