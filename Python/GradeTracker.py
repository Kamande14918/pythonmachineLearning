import csv


def save_student_grades(file_path, student_data):
    """Save student grades to a CSV file."""
    try:
        with open(file_path, mode='a', newline='') as file:  # 'a' mode appends data
            writer = csv.writer(file)
            writer.writerow(student_data)
        print(f"Student data for {student_data[0]} has been saved.")
    except Exception as e:
        print(f"An error occurred while saving student data: {e}")


def retrieve_all_grades(file_path):
    """Retrieve all student grades from a CSV file."""
    try:
        with open(file_path, mode='r') as file:
            reader = csv.reader(file)
            print("\nStudent Grades:")
            print(f"{'Name':<15}{'Subject':<15}{'Grade':<10}")
            print("-" * 40)
            for row in reader:
                print(f"{row[0]:<15}{row[1]:<15}{row[2]:<10}")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred while retrieving student grades: {e}")


# Main program
def main():
    file_path = "student_grades.csv"

    # Ensure the CSV file has headers if it's empty
    try:
        with open(file_path, mode='r') as file:
            pass
    except FileNotFoundError:
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Subject", "Grade"])  # Adding headers

    while True:
        print("\nStudent Grade Tracker")
        print("1. Add Student Grade")
        print("2. View All Grades")
        print("3. Exit")
        choice = input("Enter your choice (1-3): ")

        if choice == "1":
            name = input("Enter student name: ").strip()
            subject = input("Enter subject: ").strip()
            grade = input("Enter grade: ").strip()
            save_student_grades(file_path, [name, subject, grade])

        elif choice == "2":
            retrieve_all_grades(file_path)

        elif choice == "3":
            print("Exiting the program. Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()

