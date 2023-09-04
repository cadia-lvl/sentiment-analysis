from transformers import pipeline


def main():
    print(pipeline("sentiment-analysis")(__name__))


if __name__ == "__main__":
    main()
