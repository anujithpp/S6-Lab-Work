#include <iostream>
#include <vector>
#include <thread>
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

// Thread sum
void sumPart(vector<int>& arr, int start, int end, int& result)
{
    result = 0;
    for(int i = start; i < end; i++)
        result += arr[i];
}

// Thread search
void searchPart(vector<int>& arr, int start, int end, int key, bool& found)
{
    for(int i = start; i < end; i++)
    {
        if(arr[i] == key)
        {
            found = true;
            return;
        }
    }
}

int main()
{
    srand(time(0));

    int n = 1000000;
    vector<int> arr(n);

    generateArray(arr, n);

    int key = arr[rand() % n];

    int mid = n / 2;

    int sum1 = 0, sum2 = 0;
    bool found1 = false, found2 = false;

    auto start = chrono::high_resolution_clock::now();

    // Create threads
    thread t1(sumPart, ref(arr), 0, mid, ref(sum1));
    thread t2(sumPart, ref(arr), mid, n, ref(sum2));

    thread t3(searchPart, ref(arr), 0, mid, key, ref(found1));
    thread t4(searchPart, ref(arr), mid, n, key, ref(found2));

    // Join threads
    t1.join();
    t2.join();
    t3.join();
    t4.join();

    int total_sum = sum1 + sum2;
    bool found = found1 || found2;

    auto end = chrono::high_resolution_clock::now();

    auto time_taken = chrono::duration_cast<chrono::milliseconds>(end - start);

    cout << "Threaded Sum: " << total_sum << endl;
    cout << "Search Result: " << (found ? "Found" : "Not Found") << endl;
    cout << "Time: " << time_taken.count() << " ms" << endl;

    return 0;
}
