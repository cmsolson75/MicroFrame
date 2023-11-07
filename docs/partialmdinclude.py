from docutils.parsers.rst import Directive
from docutils import nodes
from sphinx.util.docutils import SphinxDirective


class PartialMdInclude(SphinxDirective):
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {
        'start-line': int,
        'end-line': int,
    }

    def run(self):
        filename = self.arguments[0]
        start_line = self.options.get('start-line', 0)  # default to the beginning of the file
        end_line = self.options.get('end-line', None)  # default to the end of the file

        try:
            with open(filename, 'r') as f:
                lines = f.readlines()[start_line:end_line]  # slice the file content between start and end lines
                md_content = ''.join(lines)
        except IOError as e:
            return [self.state.document.reporter.warning(
                f'Error reading file: {e}', line=self.lineno)]

        from m2r2 import M2R
        m2r = M2R()(md_content)

        # Parse the reST content
        self.state_machine.insert_input(m2r.split('\n'), filename)
        return []


def setup(app):
    app.add_directive("partialmdinclude", PartialMdInclude)
