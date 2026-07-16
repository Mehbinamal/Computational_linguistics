import sys
import re

def part_a(lines):
    """Find lines with two consecutive repeated words."""
    pattern = re.compile(r'\b([A-Za-z]+)\s+\1\b')
    results = []
    for line in lines:
        line = line.rstrip('\n')
        if pattern.search(line):
            results.append(line)
    return results

def part_b(lines):
    """Find lines that start with an integer and end with a word."""
    pattern = re.compile(r'^[0-9]+\s+.*[A-Za-z]+$')
    results = []
    for line in lines:
        line = line.rstrip('\n')
        if pattern.search(line):
            results.append(line)
    return results

def part_c(lines):
    """Find lines containing both 'grotto' and 'raven' as whole words."""
    pattern = re.compile(r'^(?=.*\bgrotto\b)(?=.*\braven\b).*$')
    results = []
    for line in lines:
        line = line.rstrip('\n')
        if pattern.search(line):
            results.append(line)
    return results

def part_d(text):
    """Extract the first word of every sentence in the whole text."""
    pattern = re.compile(r'(?:^|[.!?]\s+)\W*([A-Za-z]+)')
    return pattern.findall(text)

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    text = ''.join(lines)

    print("===== Part a: Lines with two consecutive repeated words =====")
    for line in part_a(lines):
        print(line)

    print("===== Part b: Lines starting with integer and ending with word =====")
    for line in part_b(lines):
        print(line)

    print("===== Part c: Lines containing both 'grotto' and 'raven' =====")
    for line in part_c(lines):
        print(line)

    print("===== Part d: First word of each sentence =====")
    for word in part_d(text):
        print(word)

if __name__ == "__main__":
    main()