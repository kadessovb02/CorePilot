"""Typed response contracts use HTTP mocks, never local model processes."""
import asyncio
import json

import httpx
import pytest

from corepilot.config import Settings
from corepilot.model import ModelClient, selected_tools
from corepilot.stage_submission import stage_tool


@pytest.mark.parametrize('stage,keys', [
    ('select', {'read_paths','edit_paths','new_test_paths'}),
    ('edit', {'patches','summary'}), ('tests', {'tests','summary'}),
    ('review', {'approved','summary','issues'}), ('inspect', {'answer','references'})])
def test_only_current_stage_is_advertised_with_bounded_closed_objects(stage, keys):
    tool = stage_tool(stage)
    assert tool['function']['name'] == 'submit_' + stage
    schema = tool['function']['parameters']
    assert set(schema['properties']) == keys
    assert set(schema['required']) == keys - ({'summary'} if stage in {'edit','tests'} else set())
    def bounded(node):
        if node['type'] == 'object':
            assert node['additionalProperties'] is False
            for child in node['properties'].values(): bounded(child)
        elif node['type'] == 'array':
            assert 0 < node['maxItems'] <= 20; bounded(node['items'])
        elif node['type'] == 'string':
            assert 0 < node['maxLength'] <= 64000
    bounded(schema)
    schema['properties'].clear()
    assert set(stage_tool(stage)['function']['parameters']['properties']) == keys
    assert not any(row['function']['name'].startswith('submit_') for row in selected_tools())
    with pytest.raises(ValueError): selected_tools(['submit_' + stage])


def test_native_stage_preserves_exact_malformed_arguments_for_host_validation(tmp_path):
    async def exercise():
        arguments = '{"patches": [], "patches": [], "summary": "duplicate"}'
        client = ModelClient(Settings(root=tmp_path/'state', protocol='native'))
        payloads = []
        def respond(request):
            payloads.append(json.loads(request.content))
            return httpx.Response(200, json={'choices':[{'finish_reason':'length','message':{
                'role':'assistant','content':None,'tool_calls':[{'id':'exact-native-id','type':'function',
                    'function':{'name':'submit_edit','arguments':arguments}}]}}]})
        await client.http.aclose()
        client.http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        try:
            message = await client.complete_stage('edit',[{'role':'system','content':'Synthetic stage'}])
            with pytest.raises(ValueError):
                await client.complete_stage('unknown',[{'role':'system','content':'Synthetic stage'}])
        finally: await client.close()
        assert len(payloads) == 1
        assert [tool['function']['name'] for tool in payloads[0]['tools']] == ['submit_edit']
        assert message['tool_calls'][0]['function']['arguments'] == arguments
        assert message['tool_calls'][0]['id'] == 'exact-native-id'
        assert message['_corepilot_finish_reason'] == 'length'
        assert message['_corepilot_stage_submission'] == 'edit'
    asyncio.run(exercise())
