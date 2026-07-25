"""Guards on ``analyze_neb.load_status_csv()``.

``analyze_neb`` reads ``NEB_status_csvs/`` through its own reader rather than
``config.read_status_csv_rows()``, so it needs the same tolerance for legacy
multi-line-error spillover: before status messages were sanitized on write
(``tools.csv_safe_status``), a VASP/MPI crash dump captured as an error status
spilled across physical lines. The pre-existing ``len(row) < 4`` guard drops
short fragments, but a fragment carrying >= 4 commas survives it and then hits
``int(row[0])``.

Every real NEB row is ``{job_id},{rank},{sub_band_id},"{status}"`` with three
leading integers, so requiring that is sufficient to separate real rows from
spillover.
"""
import pytest

import saddlemill.analyze_neb as an


def _write(tmp_path, raw, name="status_rank_0.csv"):
    """Write a NEB status CSV and return the directory holding it."""
    d = tmp_path / "NEB_status_csvs"
    d.mkdir(exist_ok=True)
    (d / name).write_text(raw)
    return str(d)


class TestLoadStatusCsvWellFormed:
    """Unchanged behaviour on clean input (guards against collateral damage)."""

    def test_all_rows_parsed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(an, "STATUS_CSV_DIR", _write(
            tmp_path,
            '0,0,0,"converged"\n'
            '0,0,1,"not_converged"\n'
            '1,0,0,"converged"\n'))
        sel = an.load_status_csv()
        # key is (rank, job_id)
        assert set(sel) == {(0, 0), (0, 1)}
        assert sel[(0, 0)]["all_subbands"] == {0: "converged", 1: "not_converged"}
        assert sel[(0, 0)]["selected"] == [0, 1]
        assert sel[(0, 1)]["all_subbands"] == {0: "converged"}

    def test_no_warning_when_clean(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(an, "STATUS_CSV_DIR", _write(
            tmp_path, '0,0,0,"converged"\n'))
        an.load_status_csv()
        assert "malformed" not in capsys.readouterr().out

    def test_multiple_rank_shards_merged(self, tmp_path, monkeypatch):
        d = _write(tmp_path, '0,0,0,"converged"\n', "status_rank_0.csv")
        (tmp_path / "NEB_status_csvs" / "status_rank_1.csv").write_text(
            '0,1,0,"not_converged"\n')
        monkeypatch.setattr(an, "STATUS_CSV_DIR", d)
        sel = an.load_status_csv()
        assert set(sel) == {(0, 0), (1, 0)}

    def test_analyze_statuses_filter_still_applies(self, tmp_path, monkeypatch):
        """The status-selection logic must be untouched by the guard."""
        monkeypatch.setattr(an, "STATUS_CSV_DIR", _write(
            tmp_path,
            '0,0,0,"converged_CI"\n'
            '0,0,1,"not_converged"\n'))
        monkeypatch.setattr(an, "ANALYZE_STATUSES", {"not_converged"})
        sel = an.load_status_csv()
        # Only sub-band 1 matches; sub-band 0 stays visible in all_subbands.
        assert sel[(0, 0)]["selected"] == [1]
        assert set(sel[(0, 0)]["all_subbands"]) == {0, 1}

    def test_empty_dir_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(an, "STATUS_CSV_DIR", _write(tmp_path, ""))
        assert an.load_status_csv() == {}


class TestLoadStatusCsvSpillover:
    """Legacy crash-dump spillover must be skipped, never raise."""

    def test_long_fragment_does_not_raise(self, tmp_path, monkeypatch, capsys):
        """The exact gap: a fragment with >= 4 comma-separated fields clears
        the ``len(row) < 4`` guard and used to reach ``int(row[0])``."""
        raw = (
            '0,0,0,"converged"\n'
            'Process "name", pid 123, on node c1, exited on signal 15\n'
            '1,0,0,"not_converged"\n'
        )
        monkeypatch.setattr(an, "STATUS_CSV_DIR", _write(tmp_path, raw))
        sel = an.load_status_csv()  # must not raise
        assert set(sel) == {(0, 0), (0, 1)}
        assert "skipped 1 malformed" in capsys.readouterr().out

    def test_short_fragments_still_skipped_silently(self, tmp_path, monkeypatch, capsys):
        """< 4 fields is handled by the pre-existing guard, not the new one,
        so it must not be counted as 'malformed'."""
        raw = (
            '0,0,0,"converged"\n'
            'forrtl: error (78): process killed (SIGTERM)\n'
            '--------------------------------------------\n'
        )
        monkeypatch.setattr(an, "STATUS_CSV_DIR", _write(tmp_path, raw))
        sel = an.load_status_csv()
        assert set(sel) == {(0, 0)}
        assert "malformed" not in capsys.readouterr().out

    @pytest.mark.parametrize("bad", [
        'notanint,0,0,"converged"',   # non-integer job_id
        '0,notanint,0,"converged"',   # non-integer rank
        '0,0,notanint,"converged"',   # non-integer sub_band_id
    ])
    def test_non_integer_leading_fields_skipped(self, tmp_path, monkeypatch, bad):
        monkeypatch.setattr(an, "STATUS_CSV_DIR", _write(
            tmp_path, f'0,0,0,"converged"\n{bad}\n'))
        sel = an.load_status_csv()
        assert set(sel) == {(0, 0)}
        assert sel[(0, 0)]["all_subbands"] == {0: "converged"}

    def test_realistic_multiline_dump(self, tmp_path, monkeypatch, capsys):
        """A full pre-sanitization crash dump.

        Note a quoted CSV field may legally span newlines, so newlines alone do
        *not* desynchronize the reader. The break needs an embedded ``"`` (here
        in ``Process "name"``) to close the field early; the remaining dump
        lines are then read as fresh rows, and the ``mpirun ...`` one carries
        enough commas to clear the ``len(row) < 4`` guard -- that is the row
        that used to raise.
        """
        raw = (
            '0,0,0,"converged"\n'
            '1,0,0,"error: vasp failed\n'
            'Process "name" died\n'
            'mpirun detected that one process, rank 0, on node c1, exited on signal 15\n'
            '----"\n'
            '2,0,0,"converged"\n'
        )
        monkeypatch.setattr(an, "STATUS_CSV_DIR", _write(tmp_path, raw))
        monkeypatch.setattr(an, "ANALYZE_STATUSES", None)  # inspect every row
        sel = an.load_status_csv()  # must not raise
        # All three real jobs survive; only the stray fragments are dropped.
        assert set(sel) == {(0, 0), (0, 1), (0, 2)}
        assert sel[(0, 1)]["all_subbands"][0].startswith("error: vasp failed")
        # Exactly one fragment hit the new guard; '----"' was caught by len < 4.
        assert "skipped 1 malformed" in capsys.readouterr().out
