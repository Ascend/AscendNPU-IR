import sys

class BishengIRHelper:
    print_debug = False

    @staticmethod
    def change_print_debug(print_debug=False):
        BishengIRHelper.print_debug = print_debug

    @staticmethod
    def print(*args, **kwargs):
        """Wrapper function to print to stdout and optionally to stderr."""
        print(*args, file=sys.stdout, flush=True, **kwargs)
        if BishengIRHelper.print_debug:
            print(*args, file=sys.stderr, flush=True, **kwargs)

    @staticmethod
    def print_attrs(op):
        """Print all attributes of an operation."""
        BishengIRHelper.print("\n\n\n------- Printing attributes ---------")
        BishengIRHelper.print(f"Attributes of {op.name}\n")
        if op.attributes:
            tmp = list(op.attributes)
            BishengIRHelper.print("Entire Attributes: ", tmp)
            for i, att in enumerate(tmp):
                BishengIRHelper.print(f' --> Attr {i}: {att.name} | {att.attr} - ')
        else:
            BishengIRHelper.print(f"Warning: No attributes")

        BishengIRHelper.print("\n\n\n----------------")