import re
import string
class FSA:
    def __init__(self, states, alphabet, transitions, start_state, accept_states):
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions
        self.start_state = start_state
        self.accept_states = accept_states

    def process_input(self, input_string):
        current_state = self.start_state
        for symbol in input_string:
            if symbol not in self.alphabet:
                return False
            if (current_state, symbol)  not in self.transitions:
                return False
            current_state = self.transitions[(current_state, symbol)]
        return current_state in self.accept_states
    
transitions = {}
for letter in string.ascii_letters:
    if letter in 'aeiouAEIOU':
        transitions[('q0', letter)] = 'qV'
        transitions[('qV', letter)] = 'qV'
        transitions[('qYV', letter)] = 'qV'
        transitions[('qC', letter)] = 'qV'
    else:
        transitions[('q0', letter)] = 'qC'
        transitions[('qV', letter)] = 'qC'
        transitions[('qYV', letter)] = 'qC'
        transitions[('qC', letter)] = 'qC'

transitions[('qV', 'y')] = 'qYV'
transitions[('qYV', 's')] = 'qS'
transitions[('qC','i')] = 'q1'
transitions[('q1','e')] = 'q2'
transitions[('q2','s')] = 'qS'

fsa = FSA(
    states={'q0', 'qV','qYV','qC','q1','q2','qS'},
    alphabet =  set(string.ascii_letters),
    transitions = transitions,
    start_state = 'q0',
    accept_states = {'qS'}
)

for word in ["toys", "boys", "ponies", "skies", "puppies", "bunnies","boies","toies"]:
    if fsa.process_input(word):
        print(f"{word} is accepted by the FSA.")
    else:
        print(f"{word} is not accepted by the FSA.")

pattern = re.compile(
    r'^(?:[A-Za-z]*[aeiou]ys|[A-Za-z]*[^aeiou]ies)$',
    re.IGNORECASE
)

print('--------------------------------------------------------')

for word in ["toys", "boys", "ponies", "skies", "puppies", "bunnies","boies","toies"]:
    if pattern.match(word):
        print(f"{word} matches the regex pattern.")
    else:
        print(f"{word} does not match the regex pattern.")