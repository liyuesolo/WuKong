from dataclasses import dataclass
import pathlib
import re
import sys
import textwrap
import traceback


@dataclass
class Config:
    add_temporary_array: bool = True
    generate_function: bool = True


class UsageError(BaseException):
    pass


def main(config: Config):
    def obtain_offsets(offset_arg):
        try:
            pairs = [pair.split("=")
                for pair in offset_arg.split("," if "," in offset_arg else " ")]
            offsets = {}
            for offs, var in sorted((tuple(reversed(p)) for p in pairs), reverse=True):
                offsets[int(offs)] = var
            offsets_natural_order = dict(reversed(offsets.items()))
            print("Offsets:", offsets_natural_order, file=sys.stderr)
        except Exception as e:
            print("Bad offsets", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            raise UsageError()
        return offsets, offsets_natural_order


    if len(sys.argv) < 2:
        raise UsageError()
    sf = pathlib.Path(sys.argv[1])
    if not sf.exists:
        print(f"File {sf} does not exist", file=sys.stderr)
        raise UsageError()

    match sys.argv[2:]:
        case []:
            offsets = None
            truncate = None
        case ["truncate", trunc_idx]:
            offsets = None
            truncate = trunc_idx
        case [offsets]:
            offsets, offsets_natural_order = obtain_offsets(offsets)
            truncate = None
        case [offsets, "truncate", keep_entries, input_length]:
            offsets, offsets_natural_order = obtain_offsets(offsets)
            truncate = (int(keep_entries), int(input_length))
        case [*_]:
            raise UsageError()

    with open(sf) as f:
        code = f.read()

    def set_int(values):
        return set(map(int, values))

    temporary_vars = set_int(re.findall(r"v\[(\d*?)\]", code))

    code_new = code
    if truncate is not None:
        keep_entries, input_length = truncate 
        print(f"Truncating results to first {keep_entries} entries", file=sys.stderr)
        def list_sort_int_set(values):
            return list(sorted(map(int, set(values))))
        x_indices = set_int(re.findall(r"x\[(\d*)\]", code_new))
        y_indices = set_int(re.findall(r"y\[(\d*)\]", code_new))
        y_indices_used = set_int(re.findall(r"=.*y\[(\d*)\]", code_new))
        # determine if jacobian or hessian
        n_y = len(y_indices)
        if n_y == input_length:
            print("Jacobian", file=sys.stderr)
        elif n_y == input_length**2:
            print("Hessian", file=sys.stderr)
        else:
            print("Bad number of indices", file=sys.stderr)
            return
        # find out which outputs shall be truncated
        y_indices_truncated = set(y for y in y_indices if y >= keep_entries*input_length or y % input_length >= keep_entries)
        print("Truncating y indices", y_indices_truncated, file=sys.stderr)
        print("Used y indices", y_indices_used, file=sys.stderr)

        y_indices_replace = y_indices_truncated.intersection(y_indices_used)
        # y_indices_truncated = 
        # print("Indices reused: ", y_indices_used, file=sys.stderr)
        # print("Indices truncated: ", y_indices_truncated, file=sys.stderr)
        print("Indices to be replaced: ", y_indices_replace, file=sys.stderr)
        for y_rep in y_indices_replace:
            v_rep = max(temporary_vars) + 1 if temporary_vars else 0
            temporary_vars.add(v_rep)
            code_new = code_new.replace(f"y[{y_rep}]", f"v[{v_rep}]")

        code_new = "\n".join(
            line for line in code_new.split("\n") if not set_int(re.findall(r"y\[(\d*)\]", line)).intersection(y_indices_truncated)
        )
        print("Re-numbering indices", file=sys.stderr)
        def renumberer(match):
            idx_orig = int(match.groups()[0])
            row, col = idx_orig // input_length, idx_orig % input_length
            idx_new = row * keep_entries + col
            #print(f"{idx_orig} ({row}, {col}) -> {idx_new}", file=sys.stderr)
            return f"y[{idx_new}]"
        code_new = re.sub(r"y\[(\d*)\]", renumberer, code_new)
        
        # find out which outputs to be truncated will be needed
        # add additional temporary variables
    if offsets is not None:
        print("Replacing offsets", file=sys.stderr)

        def substituter(match):
            var_orig, idx_orig = match.groups()
            idx_orig = int(idx_orig)
            for i, v in offsets.items():
                if i <= idx_orig:
                    return f"{v}[{idx_orig-i}]"
        code_new = re.sub(r"(x)\[(\d*?)\]", substituter, code_new)
    
    if config.add_temporary_array:
        print("Adding temporary array", file=sys.stderr)
        if temporary_vars:
            code_new = f'std::array<double, {max(temporary_vars)+1}> v;  // temporary variables\n' + code_new
    if config.generate_function:
        if not offsets:
            print("Missing offsets: not generating function", file=sys.stderr)
        else:
            print("Generating function", file=sys.stderr)
            arguments = ", ".join((
                *(f"const double *{var}" for var in offsets_natural_order.values()),
                "double* y"
            ))
            head = f"void func({arguments})" r" {"
            offset_comment = f"// Offsets {', '.join(f'{o}: {c}' for o, c in offsets_natural_order.items())}"
            foot = "}\n"
            body = textwrap.indent("\n".join((offset_comment, code_new)), '\t')
            code_new = "\n".join((head, body, foot))

    print("Removing blank lines", file=sys.stderr)
    code_new = "\n".join(line for line in code_new.split("\n") if line.strip())
    print(code_new)


if __name__ == '__main__':
    config = Config()
    try:
        main(config)
    except UsageError:
        print ("Usage: transform_code FILE [OFFSETS] [truncate KEEP_ENTRIES TOTAL_ENTRIES]", file=sys.stderr)
