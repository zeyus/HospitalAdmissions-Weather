from time import sleep
from ansi_escapes import ansiEscapes
from PyGnuplot import gp
from pathlib import Path
from queue import Empty
import os, sys, tempfile


PLOT_HEIGHT = 40
PLOT_WIDTH = 120

def is_gnuplot_available(gnuplot_bin = 'gnuplot') -> bool:
    return not bool(os.system(f'{gnuplot_bin} --version'))

def write_plot_data(data: list[tuple]) -> Path:
    tmp_file = Path(tempfile.mktemp())
    rows = len(data)
    with open(tmp_file, 'w') as f:
        # data is a list of columns
        # we only want the last max_rows rows
        for row in data[-rows:]:
            f.write('\t'.join(map(str, row)) + '\n')
    return tmp_file


# set multiplot layout 1,2
def prepare_plot() -> gp:
    plt = gp()
    plt.default_term = 'dumb'
    plt.c(f'set terminal dumb {PLOT_WIDTH},{PLOT_HEIGHT} enhanced ansirgb')
    
    return plt

def _write_plot_stdout(plt: gp, extra: dict[str, str] = {}) -> None:
    max_width = os.get_terminal_size().columns
    sleep(0.01)
    lines: list[str] = []
    sys.stdout.write(ansiEscapes.cursorSavePosition + ansiEscapes.cursorUp(1) + ansiEscapes.eraseLines(PLOT_HEIGHT + 1))
    while True:
        try:
            lines.append(plt.q_out.get(timeout=0.05).rstrip())  # type: ignore
        except Empty:
            break

    for i, (key, value) in enumerate(extra.items()):
        # if the line exists in lines, append to the end of the line, otherwise add a new line
        ex_txt = f' {key}: {value}'
        if i+1 < len(lines):
            if PLOT_WIDTH + len(ex_txt) >= max_width:
                ex_txt = ex_txt[:max_width - PLOT_WIDTH - 3] + '...'
            lines[i+1] += ex_txt
        else:
            if len(ex_txt) >= max_width:
                ex_txt = ex_txt[:max_width - 3] + '...'
            lines.append(ex_txt)

    sys.stdout.write('\n'.join(lines))
    sys.stdout.write(ansiEscapes.cursorRestorePosition)
    sys.stdout.flush()

def _write_plot_stderr(plt: gp, extra: dict[str, str] = {}) -> None:
    while True:
        try:
            sys.stderr.write(plt.q_err.get(timeout=0.05).rstrip() + '\n')  # type: ignore
        except Empty:
            break
    sys.stderr.flush()

def plot_running_loss(plt: gp, data_file: Path, titles: list[str] = ['train loss', 'test loss'], xpoints = 10, extra: dict[str, str] = {}, multi = False) -> gp:
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']

    if multi:
        plt.c(f'set multiplot layout 1,{len(titles)}')
        for i, title in enumerate(titles):
            plt.c(f'set title "{title}"')
            plt.c(f'set xrange [-{xpoints}:-1]')
            plt.c(f'plot "{data_file}" using 1:{i+2} with lines lc rgb "{colors[i]}" title "{title}"')
        plt.c('unset multiplot')
        _write_plot_stdout(plt, extra)
        return plt

    plt.c(f'set xrange [-{xpoints}:-1]')
    cmd = f'plot '
    for i, title in enumerate(titles):
        cmd += f'"{data_file.as_posix()}" using 1:{i+2} with lines lc rgb "{colors[i]}" title "{title}", '
    cmd = cmd[:-2]
    plt.c(cmd)
    # _write_plot_stderr(plt)
    _write_plot_stdout(plt, extra)
    return plt
