#include <iostream> // handles inputs and outputs to the console
#include <string> // required to use strings
using namespace std;

int main()
{   // must inclose function in brackets
    string word = "sentence";
    int num_letters = word.length();

    cout << "\n" << word << "\n";
    cout << "This is a " << word << ".\n";
    cout << "There are " << num_letters << " letters in \"" << word << "\".\n";
    cout << "The 4th letter in \"" << word << "\" is \"" << word.at(3) << "\".\n";
    cout << "Thank you for coming to my TED talk.\n\n";

    return 0;
}