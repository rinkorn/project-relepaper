from enum import Enum


class PDFLength(Enum):
    SHORT = "short_pdf"
    LONG = "long_pdf"


if __name__ == "__main__":
    print(PDFLength.SHORT.value)
    print(PDFLength.LONG.value)
    print(PDFLength.SHORT)
    print(PDFLength.LONG)
