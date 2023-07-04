import argparse
import textwrap

def word_wrap_text(input_file, min_line_length, max_line_length):
    # Read the input file
    with open(input_file, 'r') as f:
        content = f.read()

    content = content.replace('\n', ';')
    content = content.replace(';;', ';')
    # Word wrap the content
    wrapped_content = textwrap.fill(content, width=max_line_length)

    # Adjust line lengths based on the minimum line length
    lines = wrapped_content.split('\n')
    for i in range(len(lines)):
        if len(lines[i]) < min_line_length and not lines[i].endswith('\n'):
            # Extend the line to the minimum length
            lines[i] = lines[i].ljust(min_line_length)

    # Join the lines and save the modified content
    modified_content = '\n'.join(lines)
    output_file = input_file.replace('.txt', '_wrapped.txt')
    with open(output_file, 'w') as f:
        f.write(modified_content)

    print(f"Modified content saved to {output_file}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Wrap text for corpus entries")
    parser.add_argument('-f', '--filename', help='Input text file')
    parser.add_argument('-mn', '--min_line_length', type=int, default=15, help='Minimum line length')
    parser.add_argument('-mxl', '--max_line_length', type=int, default=120, help='Maximum line length')
    args = parser.parse_args()

    if not all([args.filename, args.min_line_length, args.max_line_length]):
        parser.print_help()
    else:
        word_wrap_text(args.filename, args.min_line_length, args.max_line_length)
