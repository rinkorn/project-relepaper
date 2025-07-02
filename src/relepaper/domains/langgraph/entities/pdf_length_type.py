from enum import Enum


class PDFLengthType(Enum):
    SHORT = "short_pdf"
    LONG = "long_pdf"


if __name__ == "__main__":
    print(PDFLengthType.SHORT.value)
    print(PDFLengthType.LONG.value)
    print(PDFLengthType.SHORT)
    print(PDFLengthType.LONG)
