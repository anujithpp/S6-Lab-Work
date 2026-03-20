#include <iostream>
#include <thread>
using namespace std;

void printNumbers(int n)
{
    cout << "First " << n << " natural numbers are:\n";
    for(int i = 1; i <= n; i++)
    {
        cout << i << " ";
    }
    cout << endl;
}

int main()
{
    int n;
    cout << "Enter value of n: ";
    cin >> n;

    // Create thread
    thread t(printNumbers, n);

    // Wait for thread to finish
    t.join();

    return 0;
}
