import json


def save_user_data(file_path,user_data):
    """Save user data to a JSON file"""
    try:
        with open(file_path,'w') as file:
            json.dump(user_data,file,indent=4)
            print(f"User data has been saved to '{file_path}'.")
    except Exception as e:
        print(f"An error occured while saving data: {e}")

def retrieve_user_data(file_path):
    """Retrieves user data from the JSON file."""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' is not in valid JSON format.")
    except Exception as e:
        print(f"An error occurred while retrieving data: {e}")
        return None


# Example usage
def main():
    file_path ="user_data.json"

    while True:
        print("\n1. Save User Data")
        print("2. Retrieve User Data")
        print("3. Exit")
        choice = input("Enter your choice: (1-3)")

        if choice == "1":
            name= input("Enter your name: ")
            age = input("Enter your age: ")
            email = input("Enter your email: ")

            user_data = {
                "name":name,
                "age": age,
                "email":email
            }
            save_user_data(file_path, user_data)

        elif choice == "2":
            data = retrieve_user_data(file_path)
            if data:
                print("\nRetrieved User Data: ")
                for key, value in data.items():
                    print(f"{key.capitalize()}:{value}")

        elif choice == "3":
            print("Exiting the program. Goodbye!")
            break


        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
