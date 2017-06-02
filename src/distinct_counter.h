#ifndef HYPERLOGLOG_HIP_DISTINCT_COUNTER_H_
#define HYPERLOGLOG_HIP_DISTINCT_COUNTER_H_

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cassert>
#include "dense_array.h"

namespace hyperloglog_hip {
template<typename Key, typename Hash = std::hash<Key>, int NumRegisterBits = 5>
class distinct_counter {
 public:
  typedef Key key_type;
  typedef Hash hash_type;

  distinct_counter(size_t num_bucket_bits = 12)
    : num_bucket_bits_(num_bucket_bits), M_(1 << num_bucket_bits),
      c_(0), s_(1 << num_bucket_bits) {}


  void insert(const key_type &v) {
    static constexpr uint64_t num_register_bits = NumRegisterBits;
    static constexpr uint64_t register_limit = (uint64_t(1) << num_register_bits) - 1;

    const uint64_t h = hash_(v) * magic1() + magic2();
    const uint64_t h0 = h & ((uint64_t(1) << num_bucket_bits_) - 1);
    const uint64_t h1 = h >> num_bucket_bits_;

    const uint64_t b_old = M_.get(h0);
    const uint64_t b_new = h1 == 0 ? register_limit :
        std::min(register_limit, uint64_t(1 + __builtin_ctzl(h1)));

    if (b_new > b_old) {
      M_.set(h0, b_new);
      c_ += 1.0 / (s_ / (uint64_t(1) << num_bucket_bits_));
      s_ -= 1.0 / (uint64_t(1) << b_old);
      if (b_new < register_limit) {
        s_ += 1.0 / (uint64_t(1) << b_new);
      }
    }
  }

  void merge(distinct_counter& other) {
    static constexpr uint64_t num_register_bits = NumRegisterBits;
    static constexpr uint64_t register_limit = (uint64_t(1) << num_register_bits) - 1;

    assert(num_bucket_bits_ == other.num_bucket_bits_);

    const size_t num_buckets = 1 << num_bucket_bits_;

    for (size_t r = 0; r < num_buckets; ++r) {
        const uint64_t b = M_.get(r);
        const uint64_t b_other = other.M_.get(r);

        if (b < b_other) {
          c_ += 1.0 / (s_/num_buckets);
          s_ -= 1.0/(1 << b);
          const uint64_t orred = M_.get(r) | b_other;
          M_.set(r, orred);
          if (b_other < register_limit){
            s_ += 1.0/(1 << b_other);
          }
        }
    }
  }

  size_t count() const {
    return round(c_);
  }

 private:
  const size_t num_bucket_bits_;
  dense_array<NumRegisterBits> M_;
  double c_, s_;
  hash_type hash_;

  static constexpr uint64_t magic1() {
    return 9223372036854775837ULL;
  }

  static constexpr uint64_t magic2() {
    return 1234567890123456789ULL;
  }
};
}  // namespace hyperloglog_hip

#endif  // HYPERLOGLOG_HIP_DISTINCT_COUNTER_H_
