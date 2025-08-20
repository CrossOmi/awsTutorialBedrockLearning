import os
import json
import uuid
import boto3
import streamlit as st
from dotenv import load_dotenv
from botocore.exceptions import ClientError
from botocore.eventstream import EventStreamError

def initialize_session():
    """セッションの初期設定を行う"""
    if "client" not in st.session_state:
        # Bedrockへの接続設定
        st.session_state.client = boto3.client("bedrock-agent-runtime", region_name="us-east-1")

    if "session_id" not in st.session_state:
        # Bedrock Agentとの一連の会話を管理するセッションIDを設定
        # AWS Bedrock Agent側が「どのユーザーとの会話の続きか」を識別するために使われる
        st.session_state.session_id = str(uuid.uuid4())

    if "messages" not in st.session_state:
        # セッション内でのユーザーとAIの会話を初期化
        st.session_state.messages = []

    if "last_prompt" not in st.session_state:
        # 今回のセッションのプロンプトをNoneで初期化
        st.session_state.last_prompt = None

    return st.session_state.client, st.session_state.session_id, st.session_state.messages

def display_chat_history(messages):
    """チャット履歴を表示する"""
    st.title("わが家のAI技術顧問")
    st.text("画面下部のチャットボックスから何でも質問してね！")

    # チャット履歴をロール（ユーザーかAIか）によって吹き出しを変えて、会話内容をマークダウンで表示する
    for message in messages:
        with st.chat_message(message['role']):
            st.markdown(message['text'])

def display_observation_details(observation):
    """
    'observation' の内容を表示する関数（observationの存在は前提とする）
    """
    obs_type = observation.get("type")

    if obs_type == "KNOWLEDGE_BASE":
        with st.expander("🔍 ナレッジベースから検索結果を取得しました", expanded=False):
            kb_output = observation.get("knowledgeBaseLookupOutput", {})
            references = kb_output.get("retrievedReferences", [])

            if references:
                st.json(references)
            else:
                st.info("関連するドキュメントは見つかりませんでした。")

    elif obs_type == "AGENT_COLLABORATOR":
        ac_output = observation.get("agentCollaboratorInvocationOutput", {})
        agent_name = ac_output.get("agentCollaboratorName", "不明なエージェント")

        with st.expander(f"🤖 サブエージェント「{agent_name}」から回答を取得しました", expanded=True):
            output_text = ac_output.get("output", {}).get("text")

            if output_text:
                st.markdown(output_text)
            else:
                st.info("サブエージェントからのテキスト出力はありませんでした。")

def handle_trace_event(event):
    """トレースイベントの処理を行う"""
    if "orchestrationTrace" not in event["trace"]["trace"]:
        return

    trace = event["trace"]["trace"]["orchestrationTrace"]

    # 「モデル入力」トレースの表示
    if "modelInvocationInput" in trace:
        with st.expander("🤔 思考中…", expanded=False):
            input_trace = trace["modelInvocationInput"]["text"]
            try:
                json_object = st.json(json.loads(input_trace))
                # ターミナルにJSONオブジェクトを出力
                print("--- JSONとして処理 ---")
                print(json_object)
            except:
                # ターミナルに元の文字列を出力
                print("--- 文字列として処理 ---")
                print(input_trace)
                st.write(input_trace)

    # 「モデル出力」トレースの表示
    if "modelInvocationOutput" in trace:
        output_trace = trace["modelInvocationOutput"]["rawResponse"]["content"]
        with st.expander("💡 思考がまとまりました", expanded=False):
            try:
                thinking = json.loads(output_trace)["content"][0]["text"]
                if thinking:
                    st.write(thinking)
                else:
                    st.write(json.loads(output_trace)["content"][0])
            except:
                st.write(output_trace)

    # 「根拠」トレースの表示
    if "rationale" in trace:
        with st.expander("✅ 次のアクションを決定しました", expanded=True):
            st.write(trace["rationale"]["text"])

    # 「ツール呼び出し」トレースの表示
    if "invocationInput" in trace:
        invocation_type = trace["invocationInput"]["invocationType"]

        if invocation_type == "AGENT_COLLABORATOR":
            agent_name = trace["invocationInput"]["agentCollaboratorInvocationInput"]["agentCollaboratorName"]
            with st.expander(f"🤖 サブエージェント「{agent_name}」を呼び出し中…", expanded=True):
                st.write(trace["invocationInput"]["agentCollaboratorInvocationInput"]["input"]["text"])

        elif invocation_type == "KNOWLEDGE_BASE":
            with st.expander("📖 ナレッジベースを検索中…", expanded=False):
                st.write(trace["invocationInput"]["knowledgeBaseLookupInput"]["text"])

        elif invocation_type == "ACTION_GROUP":
            with st.expander("💻 Lambdaを実行中…", expanded=False):
                st.write(trace['invocationInput']['actionGroupInvocationInput'])

    # 「観察」トレースの表示
    if "observation" in trace:
        # display_observation_detailsを呼び出す際に、
        # observationの中身だけを渡す
        display_observation_details(trace["observation"])

def invoke_bedrock_agent(client, session_id, prompt):
    """Bedrockエージェントを呼び出す"""
    load_dotenv()
    return client.invoke_agent(
        agentId=os.getenv("AGENT_ID"),
        agentAliasId=os.getenv("AGENT_ALIAS_ID"),
        sessionId=session_id,
        enableTrace=True,
        inputText=prompt,
    )

def handle_agent_response(response, messages):
    """エージェントのレスポンスを処理する"""
    with st.chat_message("assistant"):
        for event in response.get("completion"):
            print(f"responseのcompletion: {event}")
            if "trace" in event:
                handle_trace_event(event)

            if "chunk" in event:
                answer = event["chunk"]["bytes"].decode()
                st.write(answer)
                messages.append({"role": "assistant", "text": answer})

def show_error_popup(exeption):
    """エラーポップアップを表示する"""
    if exeption == "dependencyFailedException":
        error_message = "【エラー】ナレッジベースのAurora DBがスリープしていたようです。数秒おいてから、ブラウザをリロードして再度お試しください🙏"
    elif exeption == "throttlingException":
        error_message = "【エラー】Bedrockのモデル負荷が高いようです。1分待ってから、ブラウザをリロードして再度お試しください🙏（改善しない場合は、モデルを変更するか[サービスクォータの引き上げ申請](https://aws.amazon.com/jp/blogs/news/generative-ai-amazon-bedrock-handling-quota-problems/)を実施ください）"
    st.error(error_message)

def main():
    """メインのアプリケーション処理"""
    client, session_id, messages = initialize_session()
    display_chat_history(messages)

    # チャット入力欄を表示し、ユーザーが入力した文字列をpromptに代入
    if prompt := st.chat_input("例：画像入り資料を使ったRAGアプリを作るにはどうすればいい？"):
        # ユーザーが入力したプロンプトを会話履歴リストに追加
        messages.append({"role": "human", "text": prompt})
        # 画面にユーザーの新しいチャット吹き出しを表示し、今回のプロンプトをそこに表示する
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            response = invoke_bedrock_agent(client, session_id, prompt)
            print(f"response: {response}")
            # ログの内容
            # response: {'ResponseMetadata': {'RequestId': '5e38f7a6-40d2-41a3-94da-96b376f8ba49', 'HTTPStatusCode': 200, 'HTTPHeaders': {'date': 'Tue, 19 Aug 2025 05:16:50 GMT', 'content-type': 'application/vnd.amazon.eventstream', 'transfer-encoding': 'chunked', 'connection': 'keep-alive', 'x-amzn-requestid': '5e38f7a6-40d2-41a3-94da-96b376f8ba49', 'x-amz-bedrock-agent-session-id': 'f1b7feaf-af71-43cb-943c-e8f98cbf00d6', 'x-amzn-bedrock-agent-content-type': 'application/json'}, 'RetryAttempts': 0}, 'contentType': 'application/json', 'sessionId': 'f1b7feaf-af71-43cb-943c-e8f98cbf00d6', 'completion': <botocore.eventstream.EventStream object at 0x0000023477CC7230>}
            handle_agent_response(response, messages)

        except (EventStreamError, ClientError) as e:
            if "dependencyFailedException" in str(e):
                show_error_popup("dependencyFailedException")
            elif "throttlingException" in str(e):
                show_error_popup("throttlingException")
            else:
                raise e

if __name__ == "__main__":
    main()