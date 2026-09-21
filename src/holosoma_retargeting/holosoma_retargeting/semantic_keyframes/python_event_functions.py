"""Parse restricted Python source into the existing trajectory predicate program.

No model source is executed. Whitelisted AST nodes preserve NumPy comparison and
bitwise boolean semantics; the existing executor still enforces contact rules.
"""
import ast
import io
import math
import re
import token
import tokenize


class PythonFunctionError(ValueError):
    def __init__(self, message, node=None, path='$', error_code='unsupported_python'):
        super().__init__(message)
        self.node, self.path, self.error_code = node, path, error_code


def strip_python_fence(raw):
    """Remove one optional Markdown fence without otherwise rewriting source."""
    source = raw.strip()
    if source.startswith('```'):
        match = re.fullmatch(r'```(?:python|py)?\s*\n([\s\S]*?)\n```', source)
        if match:
            source = match.group(1)
    return source


def syntax_repairable(errors):
    """Route only the known NumPy precedence failure to the syntax agent."""
    return bool(errors) and all(
        error.get('error_code') == 'ambiguous_bitwise_comparison' for error in errors)


def _source_tokens(raw):
    """Return normalized semantic tokens plus their lossless token stream."""
    source = strip_python_fence(raw)
    ignored_types = {
        token.ENDMARKER, token.INDENT, token.DEDENT, token.NEWLINE, tokenize.NL,
        token.ENCODING, token.COMMENT,
    }
    result, semantic_record_indices = [], []
    try:
        records = list(tokenize.generate_tokens(io.StringIO(source).readline))
        for record_index, item in enumerate(records):
            if item.type in ignored_types:
                continue
            value = item.string
            if item.type in (token.STRING, token.NUMBER):
                try:
                    value = repr(ast.literal_eval(value))
                except (SyntaxError, ValueError):
                    pass
            result.append((item.type, value))
            semantic_record_indices.append(record_index)
    except (IndentationError, SyntaxError, tokenize.TokenError) as exc:
        raise ValueError('Cannot establish protected tokens: ' + str(exc)) from exc
    return result, records, semantic_record_indices


def validate_syntax_only_repair(original, repaired, catalog, signals, schema):
    """Accept only inserted atom parentheses followed by strict parser success."""
    try:
        before, _, _ = _source_tokens(original)
        after, repaired_records, semantic_record_indices = _source_tokens(repaired)
    except ValueError as exc:
        return None, [dict(path='$', reason=str(exc))]
    original_index = 0
    inserted = []
    for repaired_index, item in enumerate(after):
        if original_index < len(before) and item == before[original_index]:
            original_index += 1
        elif item == (token.OP, '(') or item == (token.OP, ')'):
            inserted.append((repaired_index, item[1]))
        else:
            original_index = -1
            break
    if original_index != len(before):
        return None, [dict(
            path='$',
            reason=('Syntax repair changed protected source tokens or moved them instead of only inserting '
                    'parentheses; the response was discarded'),
        )]
    stack, composite_parentheses = [], set()
    for repaired_index, value in inserted:
        if value == '(':
            stack.append(repaired_index)
            continue
        if not stack:
            return None, [dict(path='$', reason='Syntax repair inserted unbalanced parentheses')]
        start = stack.pop()
        if any(item == (token.OP, '&') or item == (token.OP, '|')
               for item in after[start+1:repaired_index]):
            composite_parentheses.update((start, repaired_index))
    if stack:
        return None, [dict(path='$', reason='Syntax repair inserted unbalanced parentheses')]
    parsed, errors = parse_python_functions(repaired, catalog, signals, schema)
    if errors or not composite_parentheses:
        return parsed, errors
    remove_records = {semantic_record_indices[index] for index in composite_parentheses}
    without_composite = tokenize.untokenize([
        (item.type, item.string) for index, item in enumerate(repaired_records)
        if index not in remove_records
    ])
    baseline, baseline_errors = parse_python_functions(
        without_composite, catalog, signals, schema)
    if baseline_errors or baseline != parsed:
        return None, [dict(
            path='$',
            reason=('Syntax repair regrouped boolean operators; composite parentheses are allowed '
                    'only when removing them produces the identical compiled program'),
        )]
    return parsed, []


def parse_python_functions(raw, catalog, signals, schema):
    source = strip_python_fence(raw)
    expected = {event['action'] for event in catalog['events']}
    errors, functions, seen = [], [], set()

    def fail(message, node, path, error_code='unsupported_python'):
        raise PythonFunctionError(message, node, path, error_code)

    def mapping(node, path):
        if not isinstance(node, ast.Dict):
            fail('Expected a Python dictionary literal', node, path)
        result = {}
        for key, value in zip(node.keys, node.values):
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                fail('Dictionary keys must be literal strings; unpacking is not supported', node, path)
            if key.value in result:
                fail('Duplicate dictionary key: '+key.value, key, path)
            result[key.value] = value
        return result

    def literal(node, path):
        if isinstance(node, ast.Constant) and type(node.value) in (str, int, float, bool, type(None)):
            value = node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = literal(node.operand, path)
            if type(value) not in (int, float):
                fail('Unary sign requires a literal number', node, path)
            value = -value if isinstance(node.op, ast.USub) else value
        else:
            fail('Expected a literal value, not a call, variable or expression', node, path)
        if type(value) in (int, float) and (abs(value) > 1e100 or not math.isfinite(value)):
            fail('Value must be a finite physical number', node, path)
        return value

    def predicate(node, path, depth=0):
        if depth > 8:
            fail('Predicate nesting exceeds 8', node, path)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitAnd, ast.BitOr)):
            op = 'all' if isinstance(node.op, ast.BitAnd) else 'any'
            return {op: [predicate(node.left, path, depth+1), predicate(node.right, path, depth+1)]}
        if isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1:
            left = node.left
            slice_node = left.slice if isinstance(left, ast.Subscript) else None
            if isinstance(slice_node, ast.Index):  # Python 3.8 AST compatibility.
                slice_node = slice_node.value
            if not (isinstance(left, ast.Subscript) and isinstance(left.value, ast.Name)
                    and left.value.id == 'signals' and isinstance(slice_node, ast.Constant)
                    and isinstance(slice_node.value, str)):
                fail('Comparison must use signals["available_name"] on its left; no frame indexing', left, path)
            name = slice_node.value
            if name not in signals:
                fail('Unknown signal '+repr(name)+'; available: '+', '.join(sorted(signals)), left, path)
            ops = {ast.Gt: 'gt', ast.GtE: 'ge', ast.Lt: 'lt', ast.LtE: 'le'}
            if type(node.ops[0]) not in ops:
                fail('Comparison must use >, >=, < or <=', node, path)
            number = literal(node.comparators[0], path)
            if type(number) not in (int, float):
                fail('Comparison threshold must be a literal number', node.comparators[0], path)
            return dict(signal=name, op=ops[type(node.ops[0])], value=number)
        has_bitwise = any(isinstance(child, ast.BinOp) and isinstance(child.op, (ast.BitAnd, ast.BitOr))
                          for child in ast.walk(node))
        error_code = ('ambiguous_bitwise_comparison'
                      if isinstance(node, ast.Compare) and has_bitwise else 'unsupported_predicate')
        fail('Use (signals["name"] < NUMBER) with parenthesized comparisons joined by & or |; '
             'and/or, chained comparisons, calls and JSON predicate objects are unsupported',
             node, path, error_code)

    try:
        if len(source) > 100000:
            raise PythonFunctionError('Python source exceeds 100000 characters')
        tree = ast.parse(source)
        if sum(1 for _ in ast.walk(tree)) > 10000:
            raise PythonFunctionError('Python source AST is too large')
        for node in tree.body:
            path = getattr(node, 'name', '$')
            try:
                if not isinstance(node, ast.FunctionDef) or node.decorator_list or node.returns or node.type_comment:
                    fail('Only undecorated def EVENT_NAME(signals) functions are allowed; no imports or top-level code', node, path)
                args = node.args
                if (len(args.args) != 1 or args.args[0].arg != 'signals' or args.args[0].annotation
                        or args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg or args.defaults):
                    fail('Function signature must be def EVENT_NAME(signals)', node, path)
                if node.name not in expected or node.name in seen:
                    fail('Unknown or duplicate event function; expected exactly: '+', '.join(sorted(expected)), node, path)
                seen.add(node.name)
                body = node.body
                if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                    body = body[1:]
                if len(body) != 1 or not isinstance(body[0], ast.Return):
                    fail('Function body must contain a single return dictionary (optional docstring); no statements/calls', node, path)
                output = mapping(body[0].value, path)
                if set(output) != {'function_rationale', 'recognition'}:
                    fail('Return exactly function_rationale and recognition', body[0], path)
                recognition = mapping(output['recognition'], path+'.recognition')
                compiled = {}
                for key, expr in recognition.items():
                    location = path+'.recognition.'+key
                    if key in ('start', 'active', 'end') or (key == 'contact' and not (isinstance(expr, ast.Constant) and expr.value is None)):
                        compiled[key] = predicate(expr, location)
                    else:
                        compiled[key] = literal(expr, location)
                functions.append(dict(action=node.name, function_rationale=literal(output['function_rationale'], path+'.function_rationale'), recognition=compiled))
            except PythonFunctionError as exc:
                errors.append(dict(path=exc.path, reason=str(exc), python_line=getattr(exc.node, 'lineno', None),
                                   python_column=getattr(exc.node, 'col_offset', 0)+1,
                                   error_code=exc.error_code))
        for missing in sorted(expected-seen):
            errors.append(dict(path=missing, reason='Missing function for this fixed catalog event',
                               error_code='missing_function'))
    except SyntaxError as exc:
        errors.append(dict(path='$', reason=exc.msg, python_line=exc.lineno, python_column=exc.offset,
                           source_line=exc.text, error_code='python_syntax_error'))
    except (PythonFunctionError, ValueError, RecursionError) as exc:
        errors.append(dict(path='$', reason=str(exc),
                           error_code=getattr(exc, 'error_code', 'python_parse_failure')))
    return (None if errors else dict(schema=schema, functions=functions)), errors
