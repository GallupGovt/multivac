from multivac.src.data import get
from multivac.src.data import process


def main():
    # query apis to obtain articles
    get.main()
    # process article data for models
    process.main()


if __name__ == '__main__':
    main()