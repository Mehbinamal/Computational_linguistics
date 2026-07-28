class PluralFST:
    def __init__(self):
        self.transitions = {} # Transitions: (state, input_char) -> (next_state, output_string)

        
        for c in 'abcdefghijklmnopqrtuvwy': 
            self.transitions[(0, c)] = (0, c) # everything except x,y,z q0
            self.transitions[(1, c)] = (0, c) # if x,y,z is seen, go to q1, then back to q0 for any other char

    
        # x, s, z q0 -> q1
        for c in 'xsz':
            self.transitions[(0, c)] = (1, c)
            self.transitions[(1, c)] = (1, c)

        # Boundaries
        self.transitions[(0, '^')] = (0, '')  # epsilon
        self.transitions[(0, '#')] = (4, '')  # go to accept

        self.transitions[(1, '^')] = (2, '')  # Trigger e-insertion
        self.transitions[(1, '#')] = (4, '')  # fallback accept

        # --- State q2 (Triggered) ---
        self.transitions[(2, 's')] = (3, 'es')  # Insert 'e' + output the 's'

        # --- State q3 (E inserted) ---
        self.transitions[(3, '#')] = (4, '')  # Consume final boundary

    def run(self, input_string):
        state = 0
        output_chars = []
        
        for char in input_string:
            if (state, char) not in self.transitions:
                raise ValueError(f"Error: No transition from state {state} on input '{char}'")
            
            next_state, out_str = self.transitions[(state, char)]
            output_chars.append(out_str)
            state = next_state
        
        # check if we ended in accept state (4)
        if state != 4:
            print(f"Warning: Ended in non-accept state {state}")
        
        return ''.join(output_chars)

if __name__ == "__main__":
    fst = PluralFST()
    
    test_cases = [
        "fox^s#",
        "boy^s#",
        "bus^s#",
        "mass^s#",
        "buzz^s#",
        "car^s#"
    ]
    
    for word in test_cases:
        result = fst.run(word)
        print(f"Input:  {word}")
        print(f"Output: {result}\n")


