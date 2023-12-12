def print_grid(value_fn, states, k=None):
    states = list(states)

    if k is not None:
        print(f"k={k}")

    for i in range(len(states)):
        if not i % 4:
            print()
            print("----------------------------------")
            print("|{:6.1f} |".format(value_fn(states[i])), end='')
        else:
            print("{:6.1f} |".format(value_fn(states[i])), end='')

    print()
    print("----------------------------------")
