#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
using namespace std;

// Generate random array
void generateArray(vector<int>& arr, int n)
{
    for(int i = 0; i < n; i++)
        arr[i] = rand() % 100;
}

// Sum function
int findSum(vector<int>& arr)
{
    int sum = 0;
    for(int x : arr)
        sum += x;
    return sum;
}

// Search function
bool searchElement(vector<int>& arr, int key)
{
    for(int x : arr)
        if(x == key)
            return true;
    return false;
}

int main()
{
    srand(time(0));

    int n = 1000000;
    vector<int> arr(n);

    generateArray(arr, n);

    int key = arr[rand() % n];

    auto start = chrono::high_resolution_clock::now();

    int sum = findSum(arr);
    bool found = searchElement(arr, key);

    auto end = chrono::high_resolution_clock::now();

    auto time_taken = chrono::duration_cast<chrono::milliseconds>(end - start);

    cout << "Sequential Sum: " << sum << endl;
    cout << "Search Result: " << (found ? "Found" : "Not Found") << endl;
    cout << "Time: " << time_taken.count() << " ms" << endl;

    return 0;
}
