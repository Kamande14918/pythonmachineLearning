class Book:
    """Represents a book in the library."""
    def __init__(self, title, author, copies):
        self.title = title
        self.author = author
        self.copies = copies

    def is_available(self):
        """Check if the book is available."""
        return self.copies > 0

    def borrow(self):
        """Borrow a book (reduce the number of copies)."""
        if self.is_available():
            self.copies -= 1
            return True
        return False

    def return_book(self):
        """Return a book (increase the number of copies)."""
        self.copies += 1

    def __str__(self):
        """String representation of a book."""
        return f"Title: {self.title}, Author: {self.author}, Copies: {self.copies}"


class Library:
    """Represents the library."""
    def __init__(self):
        self.books = {}

    def add_book(self, title, author, copies):
        """Add a book to the library."""
        if title in self.books:
            self.books[title].copies += copies
        else:
            self.books[title] = Book(title, author, copies)
        print(f"Added {copies} copies of '{title}' by {author}.")

    def display_books(self):
        """Display all books in the library."""
        print("\nLibrary Catalog:")
        for book in self.books.values():
            print(book)

    def borrow_book(self, title):
        """Borrow a book from the library."""
        if title in self.books:
            if self.books[title].borrow():
                print(f"You have borrowed '{title}'.")
            else:
                print(f"Sorry, '{title}' is not available right now.")
        else:
            print(f"'{title}' is not in the library catalog.")

    def return_book(self, title):
        """Return a borrowed book to the library."""
        if title in self.books:
            self.books[title].return_book()
            print(f"'{title}' has been returned.")
        else:
            print(f"'{title}' does not belong to this library.")


class Patron:
    """Represents a library patron."""
    def __init__(self, name):
        self.name = name
        self.borrowed_books = []

    def borrow_book(self, library, title):
        """Borrow a book from the library."""
        if library.books.get(title) and library.books[title].is_available():
            library.borrow_book(title)
            self.borrowed_books.append(title)
        else:
            print(f"Sorry, {self.name}, '{title}' is not available.")

    def return_book(self, library, title):
        """Return a borrowed book to the library."""
        if title in self.borrowed_books:
            library.return_book(title)
            self.borrowed_books.remove(title)
        else:
            print(f"{self.name} does not have '{title}' to return.")

    def display_borrowed_books(self):
        """Display the list of borrowed books."""
        print(f"{self.name}'s Borrowed Books: {', '.join(self.borrowed_books) if self.borrowed_books else 'None'}")


# Main Program
if __name__ == "__main__":
    # Create a library
    library = Library()

    # Add books to the library
    library.add_book("1984", "George Orwell", 5)
    library.add_book("To Kill a Mockingbird", "Harper Lee", 3)
    library.add_book("The Great Gatsby", "F. Scott Fitzgerald", 4)

    # Display all books in the library
    library.display_books()

    # Create a patron
    patron1 = Patron("Alice")
    patron2 = Patron("Bob")

    # Patrons borrow books
    patron1.borrow_book(library, "1984")
    patron1.borrow_book(library, "To Kill a Mockingbird")
    patron2.borrow_book(library, "1984")

    # Display borrowed books for each patron
    patron1.display_borrowed_books()
    patron2.display_borrowed_books()

    # Display all books after borrowing
    library.display_books()

    # Patrons return books
    patron1.return_book(library, "1984")
    patron1.return_book(library, "The Catcher in the Rye")  # Book not borrowed
    patron2.return_book(library, "1984")

    # Display borrowed books after returning
    patron1.display_borrowed_books()
    patron2.display_borrowed_books()

    # Display all books after returning
    library.display_books()
