#include <iostream> // #include is like imports in python
#include <list>

using namespace std;

// extern is for variable sharing between files
extern int num;

string object;

// main() is where program execution begins.
int main() {
   cout << "Hello World\n"; // prints Hello World
   int num = 5;
   string object = " dogs!";
   cout << "I have " + to_string(num) + object + "\n";

   // Declare and initialize a list
   // list<int> range (0,10);
   int i = 1; // iterator
   for (i; i!=10;i++) {
   	cout << "The value of i is : " << i << endl;
   }

   return 0;
}