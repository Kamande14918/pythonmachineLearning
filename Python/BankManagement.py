class BankAccount:
    """Base class for bank account."""
    def __init__(self,account_number, account_holder,account_balance =0):
        self.account_number = account_number
        self.account_holder = account_holder
        self.balance = account_balance

    def deposit(self,amount):
        """Deposit money into account"""
        if amount > 0:
            self.balance += amount
            print(f"Deposited ${amount:.2f} into account {self.account_number}. New balance is ${self.balance:.2f}")
        else:
            print("Invalid deposit amount.")
    def withdraw(self,amount):
        """Withdraw money from account."""
        if 0 < amount <= self.balance:
            self.balance -= amount
            print(f"Withdrew ${amount:.2f} from account {self.account_number}. New balance is ${self.balance:.2f}")
        else:
            print("Insufficient funds or invalid withdrawal amount.")

    def get_balance(self):
        """Get current balance."""
        return self.balance

    def __str__(self):
        """String representation of the account."""
        return f"Account Number: {self.account_number}, Holder: {self.account_holder}, Balance: {self.balance:.2f}"

class SavingsAccount(BankAccount):
    """Savings account with interest."""
    interest_rate = 0.03 # 3% annual interest

    def apply_interest(self):
        """Apply interest to the account"""
        interest = self.balance * self.interest_rate
        self.balance += interest
        print(f"Interest of ${interest:.2f} applied to account {self.account_number}. New balance is ${self.balance:.2f}")
class CheckingAccount(BankAccount):
    """Checking account with overdraft limit"""
    overdraft_limit = 500 # $500 overdraft limit
    def withdraw(self,amount):
        """Withdraw money with overdraft protection."""
        if 0 < amount <= self.balance + self.overdraft_limit:
            self.balance -= amount
            print(f"Withdrew ${amount:.2f} from account: {self.account_number}. New balance is {self.balance:.2f}")
        else:
            print("Withdraw exceeds overdraft limit or invalid amount.")

class Bank:
    """A class to manage multiple bank accounts"""
    def __init__(self):
        self.accounts = {}

    def add_account(self,account):
        """Add a new account to the bank"""
        self.accounts[account.account_number] = account
        print(f"Account {account.account_number} created for  {account.account_holder}")

    def get_account(self,account_number):
        """Retrieve an account by account number."""
        return  self.accounts.get(account_number,None)

    def display_accounts(self):
        """Display all accounts in the bank."""
        print("\nBank Accounts:")
        for account in self.accounts.values():
            print(account)


# Main Program
if __name__ == "__main__":
    bank = Bank()

#     Create accounts
    acc1 = SavingsAccount(account_number="SA123",account_holder="Alice",account_balance=1000)
    acc2 = CheckingAccount(account_number="CA456",account_holder="Bob",account_balance=500)

# Add accounts to the bank
    bank.add_account(acc1)
    bank.add_account(acc2)
# Display all the accounts
    bank.display_accounts()


# Perform transactions
    acc1.deposit(500)
    acc1.apply_interest()
    acc1.withdraw(300)

    acc2.deposit(200)
    acc2.withdraw(1000)
    acc2.withdraw(600)

# Check balances
    print(f"\n{acc1.account_holder}'s Balance: {acc1.get_balance():.2f}")
    print(f"\n{acc2.account_holder}'s Balance: {acc2.get_balance():.2f}")

#     Display all accounts after transaction
    bank.display_accounts()



