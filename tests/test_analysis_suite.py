import json
from pathlib import Path
import pytest

from moonbroom.analysis_suite import run_suite


def test_analysis_suite_resolves_paths_resumes_and_checks_products(tmp_path):
    fixture = Path(__file__).parent / 'fixtures/M9906003_excerpt.LBL'
    job = tmp_path / 'decode.json'
    job.write_text(json.dumps(dict(task='decode', raw_paths=[str(fixture.resolve())], output_directory='results/decode')))
    pipeline = tmp_path / 'pipeline.json'
    pipeline.write_text(json.dumps(dict(output_directory='results', jobs=['decode.json'])))
    first = run_suite(pipeline)
    assert first[0]['status'] == 'completed'
    assert run_suite(pipeline)[0]['status'] == 'reused'
    (tmp_path / 'results/decode/audit.csv').write_text('changed')
    with pytest.raises(ValueError, match='Completed analysis changed'):
        run_suite(pipeline)
