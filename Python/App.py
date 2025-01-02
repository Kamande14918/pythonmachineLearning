from msilib import make_id


class Car:
    wheels = 4 #Class Variable
    def __init__(self,make,model): #Constructor
        self.make = make #Initiate variable
        self.model =model

    def start(self):
        print(f"The {self.make} {self.model} is starting!")
    def stop(self):
        print(f"The {self.make} {self.model} is stopping!")


# Creating objects (instances)
car1= Car("Toyota","Corolla")
car2 = Car("Honda","Civic")
# print(car1.wheels)
# print(car2.wheels)

# Changing the class variable
Car.wheels = 3
# print(car1.wheels)
# print(car2.wheels)
# USing methods
# car1.start()
# car2.stop()

# The constructor (__init__)
# Example: In a library system, the book class might require attributes like
# title, author and year_published at the time of creation
class Book:
    def __init__(self,title,author,year_published):
        self.title = title
        self.author = author
        self.year_published = year_published

    def info(self):
        print(f"'{self.title}' by {self.author}, published in {self.year_published}")

# Creating Object
book = Book("1984","George Orwell",1949)
# book.info() # Output

# Methods
class Math:
    @staticmethod
    def add(a,b):
        return  a+b

    @classmethod
    def info(cls):
        print("This is a Math class")

        # Using methods
# print(Math.add(10,20))
# Math.info()


# Inheritance
class Animal:
    def __init__(self,name):
        self.name = name

    def speak(self):
        print(f"{self.name} makes a sound!")

class Dog(Animal): #Dog inherits fro animal
    def speak(self):
        print(f"{self.name} barks!")

dog= Dog("Buddy")
dog.speak()

# Polymorphism
class Shape:
    def area(self):
        pass #Abstract method

class Cirle(Shape):
    def __init__(self,radius):
        self.radius = radius
    def area(self):
        return  3.142 * self.radius**2
class Rectangle(Shape):
    def __init__(self,width,height):
        self.width = width
        self.height = height

    def area(self):
        return  self.width * self.height

# Using Polymorphism
shapes = [Cirle(5),Rectangle(4,6)]
for shape in shapes:
    print(f"Area :{shape.area()}") #Calls appropriate area method