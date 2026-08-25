from __future__ import annotations
import json
from types import SimpleNamespace
import pytest
from agent.assistant_turn import INTERNAL_INSPECT_TOOL, INTERNAL_INVOKE_TOOL, ModelLedAssistantTurn, TurnValidationError, capability_catalog_authority, live_capability_catalog, live_registry_tools, normalize_native_tool_response, provider_transcript_tool_calls
from agent.capability_registry import ApprovalPolicy, CapabilityContract, CapabilityDefinition, CapabilityMode, CapabilityProvenance, CapabilityRegistry
from agent.llm.types import Response, ToolCall

def registry(calls):
    r=CapabilityRegistry()
    for cid, fields, data in (("filesystem.search",{"query":str},{"ok":True,"path":"/allowed/backup.md"}),("filesystem.read",{"path_hint":str},{"ok":True,"content":"recovery"})):
        r.register(CapabilityDefinition(capability_id=cid,description=cid,example_goals=("x",),input_contract=CapabilityContract(properties={"user_id":str,"text":str,**fields},required=("user_id","text")),output_contract=CapabilityContract(properties={"ok":bool},required=("ok",)),mode=CapabilityMode.READ_ONLY,approval_policy=ApprovalPolicy.NEVER,invocation_hook=lambda v,cid=cid,data=data:calls.append((cid,dict(v))) or data,verification_hook=lambda _:True,health_hook=lambda:(True,None)))
    return r
def invoke(cid,args,call_id="c"):
    return Response(text="",provider="ollama",model="m",tool_calls=(ToolCall(call_id,INTERNAL_INVOKE_TOOL,json.dumps({"capability_id":cid,"arguments_json":args if isinstance(args,str) else json.dumps(args)})),))

def test_two_neutral_tools_catalog_no_aliases():
    r=registry([]); tools,names=live_registry_tools(r)
    assert names=={} and {x["function"]["name"] for x in tools}=={INTERNAL_INSPECT_TOOL,INTERNAL_INVOKE_TOOL}
    assert all("invocation_name" not in x for x in live_capability_catalog(r))
def test_exact_status_style_invocation_and_validation():
    r=registry([]); a=capability_catalog_authority(r)
    assert normalize_native_tool_response(invoke("filesystem.search",{"query":"x"}),r,{},catalog_authority=a)["action"]=="invoke"
    for args in ("{","[]",{"query":7}):
        assert normalize_native_tool_response(invoke("filesystem.search",args),r,{},catalog_authority=a)["action"]=="validation_observation"
def test_stale_unknown_case_cross_turn_and_revoked_fail():
    r=registry([]); a=capability_catalog_authority(r)
    for cid,catalog in (("Filesystem.Search",a),("unknown.x",a),("filesystem.search",{})):
        with pytest.raises(TurnValidationError): normalize_native_tool_response(invoke(cid,{}),r,{},catalog_authority=catalog)
    source=r.require("filesystem.search"); r.register(CapabilityDefinition(**{**source.__dict__,"capability_id":"pack.lookup","provenance":CapabilityProvenance.PACK}))
    a=capability_catalog_authority(r); assert r.unregister_external("pack.lookup")
    with pytest.raises(TurnValidationError): normalize_native_tool_response(invoke("pack.lookup",{"query":"x"}),r,{},catalog_authority=a)
def test_mutation_retains_runtime_approval_boundary():
    r=registry([]); s=r.require("filesystem.search")
    r.register(CapabilityDefinition(**{**s.__dict__,"capability_id":"filesystem.create","mode":CapabilityMode.MUTATING,"approval_policy":ApprovalPolicy.REQUIRED}))
    assert normalize_native_tool_response(invoke("filesystem.create",{"query":"x"}),r,{},catalog_authority=capability_catalog_authority(r))["calls"][0]["capability_id"]=="filesystem.create"
def test_generic_provider_transcript_and_dependent_read():
    calls=[]; r=registry(calls); response=invoke("filesystem.search",{"query":"x"},"same")
    assert provider_transcript_tool_calls(response,{"calls":[{"call_id":"same","capability_id":"filesystem.search"}]})[0].name==INTERNAL_INVOKE_TOOL
    replies=iter([invoke("filesystem.search",{"query":"backup"},"search"),invoke("filesystem.read",{"path_hint":"/allowed/backup.md"},"read"),Response(text="grounded",provider="ollama",model="m")])
    class P:
        def chat(self,*_,**__): return next(replies)
    class C:
        config=SimpleNamespace(llm_provider="ollama",ollama_model="ollama:m")
        def provider_for_id(self,_): return P()
    out=ModelLedAssistantTurn(registry=r,llm_client=C(),invoke=lambda cid,v:r.invoke(cid,v),available=lambda:True).run(user_text="find read",user_id="u")
    assert out.data["assistant_turn"]["outcome"]=="respond" and [x[0] for x in calls]==["filesystem.search","filesystem.read"]
