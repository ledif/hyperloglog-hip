#include <iostream>
#include <string>
#include "distinct_counter.h"

int main() {
  {
    // Integers
    hyperloglog_hip::distinct_counter<int> h;
    h.insert(1);
    h.insert(10);
    h.insert(5);
    h.insert(1);
    h.insert(3);
    h.insert(10);
    std::cout << h.count() << std::endl;  // #{1, 3, 5, 10} = 4
  }

  {
    // Strings
    hyperloglog_hip::distinct_counter<std::string> h;
    std::cout << h.count() << std::endl;  // #{} = 0
    h.insert("akiba");
    h.insert("iwi");
    h.insert("akiba");
    h.insert("iwi");
    h.insert("akiba");
    h.insert("iwi");
    h.insert("akiba");
    std::cout << h.count() << std::endl;  // #{akiba, iwi} = 2
    h.insert("takuya");
    std::cout << h.count() << std::endl;  // #{akiba, iwi, takuya} = 3
  }

  {
    // Merge
    hyperloglog_hip::distinct_counter<int> h0;
    hyperloglog_hip::distinct_counter<int> h1;
    h0.insert(1);
    h0.insert(10);
    h0.insert(5);

    h1.insert(1);
    h1.insert(3);
    h1.insert(10);

    h0.merge(h1);

    std::cout << h0.count() << std::endl;  // #{1, 3, 5, 10} = 4
  }
  return 0;
}
