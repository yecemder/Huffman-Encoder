"""
future notes: find way to estimate best chunking for frequency counting?
estimate final string output size, minimize? tree size shouldn't matter
too much. use gradient descent for estimate if fast enough.
for now, use single byte.

process:
count char (chunk) freqs with dict, convert to list of (chunk, freq)
heapify list into min-heap based on freq
build tree
traverse tree (stack top-down) and build index for encoding as (dict? chunk: bin)
convert to canonical huffman codes for smaller tree storage
encode text into binary with hex
encode tree:
- byte length of chunks when in bin form, stored as 16 bit unsigned int
    i.e. width of original chars or chunks (1 for char, 2 for 2-char chunks, etc)
- tree entries as list of (chunk, code length) pairs (canonical huffman)

output tree and text into bin

header needs:
- chunk size

traverse binary (decode bin string that python gives as you go, convert to bits?): 
- decode and reconstruct tree
- build tree in program
- read in data (stream to file?)


What data storage to use for char counts?
linked list: requires traversal for insertion, no move ops
array: traverses anyways to find spot for insertion, move ops
bin min-heap: o(1) of get min value, o(log n) insertion
dict -> linked list of leaf nodes -> build tree
tree build: append to never change indices, root node ends up at
end of list, use len(list) to get root node, traverse by content
"""
from __future__ import annotations
import heapq as hq
from itertools import count
from os import listdir
from pathlib import Path

class Node:
    def __init__(self, children: list[int] | None = None, contents: bytes | None = None, index: int = -1) -> None:
        # Either empty or a list of length 2 with integers, used as indices to the children.
        self.children = children
        self.contents = contents  # Contains the bytes if this is a leaf node.
        self.index = index

    def is_leaf(self) -> bool:
        return self.children is None
    
    # def __lt__(self, other: Node) -> bool:
    #     return id(self) < id(other)
    
    def __str__(self) -> str:
        if self.children:
            return f"C({self.children[0]},{self.children[1]})"
        elif self.contents:
            return f"L({self.contents})"
        else:
            return "Empty Node"

def count_freqs(data: bytes, chunk_size: int = 1) -> dict[bytes, int]:
    frequencies: dict[bytes, int] = {}
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        # if chunk in frequencies:
        #     frequencies[chunk] += 1
        # else:
        #     frequencies[chunk] = 1
        frequencies[chunk] = frequencies.get(chunk, 0) + 1
    return frequencies

def char_freq_to_tree(vals: dict[bytes, int]) -> list[Node]:
    min_heap: list[tuple[int, int, Node]] = []  # Note the order is flipped so heap can use the int freq.
    counter = count()  # Unique sequence count to avoid comparison issues in heap
    for chunk, freq in vals.items():
        hq.heappush(min_heap, (freq, next(counter), Node(contents=chunk)))
    
    tree: list[Node] = []
    left_index: int = 0
    right_index: int = 0
    
    while len(min_heap) > 1:
        left = hq.heappop(min_heap)
        right = hq.heappop(min_heap)
        
        if left[2].is_leaf(): 
            left_index = len(tree)
            tree.append(left[2])
        else:
            left_index = left[2].index

        if right[2].is_leaf(): 
            right_index = len(tree)
            tree.append(right[2])
        else:
            right_index = right[2].index
        
        # Construct our new node and add to our tree
        new_node = Node(children=[left_index, right_index], index=len(tree))
        tree.append(new_node)
        
        # Return the new node to our heap
        hq.heappush(min_heap, (left[0] + right[0], next(counter), new_node))
    
    return tree

def get_char_codes(tree: list[Node]) -> dict[bytes, str]:
    stack = []
    output: dict[bytes, str] = {}
    stack.append(['', tree[-1]])
    while len(stack) > 0:
        current = stack.pop()
        if current[1].is_leaf():
            output[current[1].contents] = current[0]
        else:
            stack.append([current[0] + '0', tree[current[1].children[0]]])
            stack.append([current[0] + '1', tree[current[1].children[1]]])

    return output

def char_code_canonical(codes: dict[bytes, str]) -> tuple[dict[bytes, str], list[tuple[bytes, int]]]:
    # Return the canonical codes (for mapping) and the list of code lengths (for later storage)
    codes_list = list(codes.items())
    # Sort by length, then lexicographically
    codes_list.sort(key=lambda x: (len(x[1]), x[0]))
    canonical_codes: dict[bytes, str] = {}
    code_lengths: list[tuple[bytes, int]] = []

    for i, (chunk, code) in enumerate(codes_list):
        canonical_codes[chunk] = code
        code_lengths.append((chunk, len(code)))

    return canonical_codes, code_lengths

def read_file_as_bin(path: Path) -> bytes:
    with open(path, mode="rb") as file:
        contents = file.read()
    return contents

input_file = Path(__file__).parent / "the_input.txt"

def main():
    print(listdir())
    binary = read_file_as_bin(input_file)
    print("binary:", binary)
    freqs = count_freqs(binary)
    print("freqs:", freqs)
    out = char_freq_to_tree(freqs)
    for i in range(len(out)):
        print(i, ":", out[i])
    codes = get_char_codes(out)
    print("codes:")
    for k, v in codes.items():
        print(k, "->", v)
    print("-"*20)
    canonical_codes, code_lengths = char_code_canonical(codes)
    print("canonical codes:")
    for k, v in canonical_codes.items():
        print(k, "->", v)
    print("code lengths:")
    for k, v in code_lengths:
        print(k, "->", v)

if __name__ == "__main__":
    main()
