import pytest

from cog_safe_push.exceptions import TestCaseFailedError
from cog_safe_push.output_checkers import (
    JqQueryChecker,
)


@pytest.mark.asyncio
async def test_jq_query_checker_basic_equality():
    checker = JqQueryChecker(query='.status == "success"')
    output = {"status": "success", "data": "test"}
    await checker(output, None)


@pytest.mark.asyncio
async def test_jq_query_checker_basic_equality_fails():
    checker = JqQueryChecker(query='.status == "success"')
    output = {"status": "failure", "data": "test"}
    with pytest.raises(TestCaseFailedError, match="returned falsy value"):
        await checker(output, None)


@pytest.mark.asyncio
async def test_jq_query_checker_numeric_comparison():
    checker = JqQueryChecker(query=".confidence > 0.8")
    output = {"confidence": 0.9, "result": "good"}
    await checker(output, None)


@pytest.mark.asyncio
async def test_jq_query_checker_numeric_comparison_fails():
    checker = JqQueryChecker(query=".confidence > 0.8")
    output = {"confidence": 0.5, "result": "bad"}
    with pytest.raises(TestCaseFailedError, match="returned falsy value"):
        await checker(output, None)


@pytest.mark.asyncio
async def test_jq_query_checker_array_length():
    checker = JqQueryChecker(query=".results | length == 5")
    output = {"results": [1, 2, 3, 4, 5]}
    await checker(output, None)


@pytest.mark.asyncio
async def test_jq_query_checker_array_length_fails():
    checker = JqQueryChecker(query=".results | length == 5")
    output = {"results": [1, 2, 3]}
    with pytest.raises(TestCaseFailedError, match="returned falsy value"):
        await checker(output, None)


@pytest.mark.asyncio
async def test_jq_query_checker_multiple_conditions():
    checker = JqQueryChecker(
        query='.status == "success" and .confidence > 0.8 and (.results | length) > 0'
    )
    output = {"status": "success", "confidence": 0.9, "results": [1, 2, 3]}
    await checker(output, None)


@pytest.mark.asyncio
async def test_jq_query_checker_multiple_conditions_fails():
    checker = JqQueryChecker(
        query='.status == "success" and .confidence > 0.8 and (.results | length) > 0'
    )
    output = {"status": "success", "confidence": 0.9, "results": []}
    with pytest.raises(TestCaseFailedError, match="returned falsy value"):
        await checker(output, None)


@pytest.mark.asyncio
async def test_jq_query_checker_nested_fields():
    checker = JqQueryChecker(query=".metadata.author and .metadata.version")
    output = {"metadata": {"author": "test", "version": "1.0"}, "data": "content"}
    await checker(output, None)


@pytest.mark.asyncio
async def test_jq_query_checker_nested_fields_fails():
    checker = JqQueryChecker(query=".metadata.author and .metadata.version")
    output = {"metadata": {"author": "test"}, "data": "content"}
    with pytest.raises(TestCaseFailedError, match="returned falsy value"):
        await checker(output, None)


@pytest.mark.asyncio
async def test_jq_query_checker_all_operator():
    checker = JqQueryChecker(query=".predictions | all(.[]; .score > 0.5)")
    output = {"predictions": [{"score": 0.6}, {"score": 0.7}, {"score": 0.8}]}
    await checker(output, None)


@pytest.mark.asyncio
async def test_jq_query_checker_all_operator_fails():
    checker = JqQueryChecker(query=".predictions | all(.[]; .score > 0.5)")
    output = {"predictions": [{"score": 0.6}, {"score": 0.3}, {"score": 0.8}]}
    with pytest.raises(TestCaseFailedError, match="returned falsy value"):
        await checker(output, None)


@pytest.mark.asyncio
async def test_jq_query_checker_any_operator():
    checker = JqQueryChecker(query='.results | any(.[]; .category == "animal")')
    output = {
        "results": [
            {"category": "plant"},
            {"category": "animal"},
            {"category": "mineral"},
        ]
    }
    await checker(output, None)


@pytest.mark.asyncio
async def test_jq_query_checker_any_operator_fails():
    checker = JqQueryChecker(query='.results | any(.[]; .category == "animal")')
    output = {"results": [{"category": "plant"}, {"category": "mineral"}]}
    with pytest.raises(TestCaseFailedError, match="returned falsy value"):
        await checker(output, None)


@pytest.mark.asyncio
async def test_jq_query_checker_type_check():
    checker = JqQueryChecker(query='(.id | type) == "string"')
    output = {"id": "abc123", "value": 42}
    await checker(output, None)


@pytest.mark.asyncio
async def test_jq_query_checker_type_check_fails():
    checker = JqQueryChecker(query='(.id | type) == "string"')
    output = {"id": 123, "value": 42}
    with pytest.raises(TestCaseFailedError, match="returned falsy value"):
        await checker(output, None)


@pytest.mark.asyncio
async def test_jq_query_checker_with_json_string():
    checker = JqQueryChecker(query='.status == "success"')
    output = '{"status": "success", "data": "test"}'
    await checker(output, None)


@pytest.mark.asyncio
async def test_jq_query_checker_with_invalid_json_string():
    checker = JqQueryChecker(query='.status == "success"')
    output = "not valid json"
    with pytest.raises(TestCaseFailedError, match="not valid JSON"):
        await checker(output, None)


@pytest.mark.asyncio
async def test_jq_query_checker_invalid_query():
    checker = JqQueryChecker(query="invalid jq syntax ][")
    output = {"status": "success"}
    with pytest.raises(TestCaseFailedError, match="jq query error|jq execution failed"):
        await checker(output, None)


@pytest.mark.asyncio
async def test_jq_query_checker_with_error():
    checker = JqQueryChecker(query='.status == "success"')
    with pytest.raises(TestCaseFailedError, match="unexpected error"):
        await checker({"status": "success"}, "some error occurred")


@pytest.mark.asyncio
async def test_jq_query_checker_keys_validation():
    checker = JqQueryChecker(
        query='keys | length == 3 and contains(["status", "data", "timestamp"])'
    )
    output = {"status": "ok", "data": "test", "timestamp": 12345}
    await checker(output, None)


@pytest.mark.asyncio
async def test_jq_query_checker_keys_validation_fails():
    checker = JqQueryChecker(
        query='keys | length == 3 and contains(["status", "data", "timestamp"])'
    )
    output = {"status": "ok", "data": "test"}
    with pytest.raises(TestCaseFailedError, match="returned falsy value"):
        await checker(output, None)


@pytest.mark.asyncio
async def test_jq_query_checker_array_indexing():
    checker = JqQueryChecker(query='.[0].type == "start" and .[-1].type == "end"')
    output = [{"type": "start"}, {"type": "middle"}, {"type": "end"}]
    await checker(output, None)


@pytest.mark.asyncio
async def test_jq_query_checker_string_contains():
    checker = JqQueryChecker(query='.message | contains("success")')
    output = {"message": "Operation completed successfully"}
    await checker(output, None)


@pytest.mark.asyncio
async def test_jq_query_checker_string_contains_fails():
    checker = JqQueryChecker(query='.message | contains("success")')
    output = {"message": "Operation failed"}
    with pytest.raises(TestCaseFailedError, match="returned falsy value"):
        await checker(output, None)


@pytest.mark.asyncio
async def test_jq_query_checker_regex_test():
    checker = JqQueryChecker(query='.id | test("^[0-9a-f]{8}-[0-9a-f]{4}")')
    output = {"id": "12345678-abcd-1234-5678-abcdef123456"}
    await checker(output, None)


@pytest.mark.asyncio
async def test_jq_query_checker_regex_test_fails():
    checker = JqQueryChecker(query='.id | test("^[0-9a-f]{8}-[0-9a-f]{4}")')
    output = {"id": "not-a-uuid"}
    with pytest.raises(TestCaseFailedError, match="returned falsy value"):
        await checker(output, None)


@pytest.mark.asyncio
async def test_jq_query_checker_or_conditions():
    checker = JqQueryChecker(
        query='.status == "success" or .status == "completed" or .status == "done"'
    )
    output = {"status": "completed"}
    await checker(output, None)


@pytest.mark.asyncio
async def test_jq_query_checker_or_conditions_fails():
    checker = JqQueryChecker(
        query='.status == "success" or .status == "completed" or .status == "done"'
    )
    output = {"status": "failed"}
    with pytest.raises(TestCaseFailedError, match="returned falsy value"):
        await checker(output, None)


@pytest.mark.asyncio
async def test_jq_query_checker_select_filter():
    checker = JqQueryChecker(query='[.items[] | select(.price > 100)] | length > 0')
    output = {"items": [{"price": 50}, {"price": 150}, {"price": 200}]}
    await checker(output, None)


@pytest.mark.asyncio
async def test_jq_query_checker_select_filter_fails():
    checker = JqQueryChecker(query='[.items[] | select(.price > 100)] | length > 0')
    output = {"items": [{"price": 50}, {"price": 75}]}
    with pytest.raises(TestCaseFailedError, match="returned falsy value"):
        await checker(output, None)
