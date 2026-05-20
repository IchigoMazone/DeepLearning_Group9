from utils.read_webcam import read_webcam
from utils.create_csv import create_csv


if __name__ == "__main__":

    print("1. Collect data")
    print("2. Create CSV")

    choice = input("Choose: ")

    if choice == "1":
        read_webcam()

    elif choice == "2":
        create_csv()