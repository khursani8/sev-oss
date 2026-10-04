"""Polarity-flip robustness test: known-answer sentiment cases.

A model passes when every case is labeled correctly. The negation cases
are the point: classifiers that read the negated complaint word and miss
the polarity flip fail here. Run against any sentiment backend.
"""
import pytest

CASES = [
    ("Parcel saya sudah 8 hari tak sampai, saya marah!", "negative"),
    ("Servis sungguh teruk, wang saya hilang begitu sahaja.", "negative"),
    ("Terima kasih, barangan sampai dengan cepat dan selamat!", "positive"),
    ("Produk ini sangat bagus, saya berpuas hati.", "positive"),
    ("Bila parcel saya akan sampai?", "neutral"),
    ("Saya ingin menanya tentang harga bulanan.", "neutral"),
    ("Saya TIDAK marah, barang sampai dengan baik.", "positive"),
    ("Barang tak rosak, jangan risau.", "positive"),
    ("Aduan saya masih belum diselesaikan, kecewa.", "negative"),
    ("Kualiti teruk sangat, menyesal beli.", "negative"),
]


@pytest.mark.parametrize("text,expected", CASES)
def test_polarity(classify, text, expected):
    assert classify(text) == expected
