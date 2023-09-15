
template<class I>
  concept forward_iterator =
    std::input_iterator<I> &&
    std::incrementable<I> &&
    std::sentinel_for<I, I>;

template<class I>
  concept bidirectional_iterator =
    std::forward_iterator<I> &&
    requires(I i) {
      { --i } -> std::same_as<I&>;
      { i-- } -> std::same_as<I>;
    };

template<class I>
  concept random_access_iterator =
    std::bidirectional_iterator<I> &&
    std::totally_ordered<I> &&
    std::sized_sentinel_for<I, I> &&
    requires(I i, const I j, const std::iter_difference_t<I> n) {
      { i += n } -> std::same_as<I&>;
      { j +  n } -> std::same_as<I>;
      { n +  j } -> std::same_as<I>;
      { i -= n } -> std::same_as<I&>;
      { j -  n } -> std::same_as<I>;
      {  j[n]  } -> std::same_as<std::iter_reference_t<I>>;
    };





template<random_access_iterator Iter, typename Compare>
void sort(Iter first, Iter last,Compare comp)
{
    
}


void sort(random_access_iterator auto first, Iter last,Compare auto comp)
{
    
}


template<input_or_output_iterator Iter>
size_t distance(Iter first, Iter last)
{
    if constexpr(random_access_iterator<Iter>)
        return last - first;
    else
    {
        size_t result{};
        for (;first != last;++first)
            ++result;
        return result;
    }
}


template<random_access_iterator Iter>
size_t distance(Iter first, Iter last)
{
 
        return last - first;
  
}

template<input_or_output_iterator Iter>
size_t distance(Iter first, Iter last)
{

      size_t result{};
      for (;first != last;++first)
          ++result;
      return result;
    
}