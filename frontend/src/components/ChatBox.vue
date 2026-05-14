<script setup lang="ts">
import { ref, nextTick, computed } from 'vue'

interface Message {
  role: 'user' | 'assistant'
  content: string
  isStreaming?: boolean
}

const messages = ref<Message[]>([])
const inputText = ref('')
const chatContainer = ref<HTMLElement | null>(null)

const knowledgeText = ref('')
const isProcessingKnowledge = ref(false)
const knowledgeStatus = ref('')
const knowledgeReady = ref(false)

const charCount = computed(() => knowledgeText.value.length)

async function sendMessage() {
  if (inputText.value.trim()) {
    messages.value.push({ role: 'user', content: inputText.value })
    messages.value.push({ role: 'assistant', content: '', isStreaming: true })
    const assistantIndex = messages.value.length - 1
    const updateAssistantMessage = (partial: Partial<Message>) => {
      // @ts-ignore
      messages.value[assistantIndex] = {
        ...messages.value[assistantIndex],
        ...partial,
      }
    }

    const userMessage = inputText.value
    inputText.value = ''

    try {
      const response = await fetch('http://localhost:8000/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage })
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const reader = response.body?.getReader()
      const decoder = new TextDecoder('utf-8')
      let buffer = ''

      if (!reader) {
        throw new Error('ReadableStream not supported in this browser.')
      }

      while (true) {
        const { value, done } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.substring(6)
            if (data === '[DONE]') {
              updateAssistantMessage({ isStreaming: false })
            } else if (data.startsWith('ERROR:')) {
              updateAssistantMessage({ content: data, isStreaming: false })
            } else if (data !== 'None') {
              const current = messages.value[assistantIndex]
              updateAssistantMessage({ content: current?.content + data })
              await nextTick()
              scrollToBottom()
            }
          }
        }
      }
    } catch (error) {
      console.error('Error:', error)
      updateAssistantMessage({ content: `请求失败: ${error}`, isStreaming: false })
    } finally {
      updateAssistantMessage({ isStreaming: false })
    }
  }
}

async function sendKnowledge() {
  const text = knowledgeText.value.trim()
  if (!text) {
    knowledgeStatus.value = '请输入知识文本'
    return
  }

  isProcessingKnowledge.value = true
  knowledgeStatus.value = '正在处理并建立索引...'

  try {
    const response = await fetch('http://localhost:8000/knowledge/process', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content: text})
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    if (text.trim()) {
      knowledgeStatus.value = '知识库已成功建立索引'
      knowledgeReady.value = true
    } else {
      knowledgeStatus.value = '知识库已清空'
      knowledgeReady.value = false
    }
  } catch (error) {
    console.error('Knowledge processing error:', error)
    knowledgeStatus.value = `处理失败: ${error}`
  } finally {
    isProcessingKnowledge.value = false
  }
}

function scrollToBottom() {
  if (chatContainer.value) {
    chatContainer.value.scrollTo({
      top: chatContainer.value.scrollHeight,
      behavior: 'smooth'
    })
  }
}
</script>

<template>
  <div class="app-layout">
    <div class="knowledge-panel">
      <div class="panel-header">
        <h2 class="panel-title">Knowledge Base</h2>
        <p class="panel-subtitle">
          输入或粘贴文本，构建知识库供 Agent 查询
        </p>
      </div>

      <div class="textarea-wrapper">
        <textarea
          v-model="knowledgeText"
          placeholder="在此输入或粘贴知识文本..."
          :disabled="isProcessingKnowledge"
        />
      </div>

      <div class="panel-footer">
        <div class="status-row">
          <span class="char-count">{{ charCount }} 字符</span>
          <span
            v-if="knowledgeStatus"
            :class="['status-msg', { error: knowledgeStatus.includes('失败') }]"
          >
            {{ knowledgeStatus }}
          </span>
        </div>
        <button
          class="process-btn"
          :disabled="isProcessingKnowledge || !knowledgeText.trim()"
          @click="sendKnowledge"
        >
          <span v-if="isProcessingKnowledge" class="spinner"></span>
          <span v-else>Process & Index</span>
        </button>
      </div>
    </div>

    <div class="chat-panel">
      <div class="panel-header">
        <h2 class="panel-title">Agent Chat</h2>
        <p class="panel-subtitle">
          与 AI Agent 进行对话 --- 
          <strong v-if="knowledgeReady">回答来源于知识库</strong>
          <strong v-else>回答来源于模型本身</strong>
        </p>
      </div>

      <div class="messages" ref="chatContainer">
        <div
          v-for="(msg, idx) in messages"
          :key="idx"
          :class="['message-row', msg.role]"
        >
          <div class="message-bubble">
            <div class="content">{{ msg.content }}</div>
            <span v-if="msg.isStreaming" class="cursor">|</span>
          </div>
        </div>
      </div>

      <div class="input-area">
        <input
          v-model="inputText"
          placeholder="输入消息..."
          @keyup.enter="sendMessage"
        />
        <button class="send-btn" @click="sendMessage">发送</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.app-layout {
  display: flex;
  width: 100vw;
  height: 100vh;
  overflow: hidden;
  font-family: 'Crimson Pro', Georgia, serif;
  background: #f5f2ed;
}

/* ---------- Left Knowledge Panel ---------- */
.knowledge-panel {
  width: 420px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  background: #ffffff;
  border-right: 1px solid #e8e4de;
  color: #2c2a28;
  box-shadow: 4px 0 24px rgba(0, 0, 0, 0.04);
}

.panel-header {
  padding: 28px 28px 16px;
  border-bottom: 1px solid #f0ede8;
}

.panel-title {
  font-family: 'Crimson Pro', Georgia, serif;
  font-size: 1.6rem;
  font-weight: 700;
  color: #b85c38;
  margin: 0 0 6px;
  letter-spacing: 0.02em;
}

.panel-subtitle {
  font-family: 'Crimson Pro', Georgia, serif;
  font-size: 0.9rem;
  color: #8a8580;
  margin: 0;
  font-style: italic;
}

.textarea-wrapper {
  flex: 1;
  padding: 20px 28px;
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.textarea-wrapper textarea {
  width: 100%;
  height: 100%;
  resize: none;
  background: #faf8f5;
  border: 1px solid #e8e4de;
  border-radius: 10px;
  padding: 16px;
  color: #2c2a28;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 0.85rem;
  line-height: 1.6;
  outline: none;
  transition: border-color 0.3s ease, box-shadow 0.3s ease, background 0.3s ease;
}

.textarea-wrapper textarea::placeholder {
  color: #b0aba4;
}

.textarea-wrapper textarea:focus {
  background: #ffffff;
  border-color: #b85c38;
  box-shadow: 0 0 0 3px rgba(184, 92, 56, 0.1);
}

.textarea-wrapper textarea:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.panel-footer {
  padding: 16px 28px 28px;
  border-top: 1px solid #f0ede8;
}

.status-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 0.75rem;
}

.char-count {
  color: #b0aba4;
}

.status-msg {
  color: #5a7a52;
  transition: color 0.3s ease;
}

.status-msg.error {
  color: #b85c38;
}

.process-btn {
  width: 100%;
  padding: 14px 0;
  background: #b85c38;
  color: #ffffff;
  border: none;
  border-radius: 10px;
  font-family: 'Crimson Pro', Georgia, serif;
  font-size: 1rem;
  font-weight: 600;
  letter-spacing: 0.04em;
  cursor: pointer;
  transition: background 0.3s ease, transform 0.15s ease, box-shadow 0.3s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}

.process-btn:hover:not(:disabled) {
  background: #a04e2e;
  box-shadow: 0 6px 24px rgba(184, 92, 56, 0.25);
  transform: translateY(-1px);
}

.process-btn:active:not(:disabled) {
  transform: translateY(0);
}

.process-btn:disabled {
  background: #e0d8d0;
  color: #a09890;
  cursor: not-allowed;
  transform: none;
  box-shadow: none;
}

.spinner {
  width: 16px;
  height: 16px;
  border: 2px solid rgba(255, 255, 255, 0.35);
  border-top-color: #ffffff;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

/* ---------- Right Chat Panel ---------- */
.chat-panel {
  flex: 1;
  display: flex;
  flex-direction: column;
  background: #faf8f5;
  color: #2c2a28;
  min-width: 0;
}

.chat-panel .panel-header {
  border-bottom: 1px solid #f0ede8;
  background: #ffffff;
}

.messages {
  flex: 1;
  overflow-y: auto;
  padding: 24px 32px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  min-height: 0;
}

.message-row {
  display: flex;
  width: 100%;
}

.message-row.user {
  justify-content: flex-end;
}

.message-row.assistant {
  justify-content: flex-start;
}

.message-bubble {
  max-width: 70%;
  padding: 14px 18px;
  border-radius: 14px;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 0.88rem;
  line-height: 1.6;
  position: relative;
  animation: fadeInUp 0.35s ease both;
}

.message-row.user .message-bubble {
  background: #b85c38;
  color: #ffffff;
  border-bottom-right-radius: 4px;
  box-shadow: 0 4px 16px rgba(184, 92, 56, 0.18);
}

.message-row.assistant .message-bubble {
  background: #ffffff;
  color: #2c2a28;
  border-bottom-left-radius: 4px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
  border: 1px solid #f0ede8;
}

.content {
  white-space: pre-wrap;
  word-break: break-word;
  display: inline-block;
}

.cursor {
  display: inline-block;
  color: #b85c38;
  animation: blink 1s infinite;
  margin-left: 2px;
}

@keyframes blink {
  50% {
    opacity: 0;
  }
}

@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(8px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* ---------- Input Area ---------- */
.input-area {
  display: flex;
  gap: 12px;
  padding: 20px 32px 28px;
  border-top: 1px solid #f0ede8;
  background: #ffffff;
}

.input-area input {
  flex: 1;
  background: #faf8f5;
  border: 1px solid #e8e4de;
  border-radius: 10px;
  padding: 14px 18px;
  color: #2c2a28;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 0.88rem;
  outline: none;
  transition: border-color 0.3s ease, box-shadow 0.3s ease, background 0.3s ease;
}

.input-area input::placeholder {
  color: #b0aba4;
}

.input-area input:focus {
  background: #ffffff;
  border-color: #b85c38;
  box-shadow: 0 0 0 3px rgba(184, 92, 56, 0.1);
}

.send-btn {
  padding: 14px 28px;
  background: transparent;
  color: #b85c38;
  border: 1.5px solid #b85c38;
  border-radius: 10px;
  font-family: 'Crimson Pro', Georgia, serif;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.3s ease, color 0.3s ease, box-shadow 0.3s ease;
}

.send-btn:hover {
  background: #b85c38;
  color: #ffffff;
  box-shadow: 0 4px 20px rgba(184, 92, 56, 0.2);
}

/* ---------- Scrollbar Styling ---------- */
.messages::-webkit-scrollbar,
textarea::-webkit-scrollbar {
  width: 6px;
}

.messages::-webkit-scrollbar-track,
textarea::-webkit-scrollbar-track {
  background: transparent;
}

.messages::-webkit-scrollbar-thumb,
textarea::-webkit-scrollbar-thumb {
  background: #d0cbc4;
  border-radius: 3px;
}

.messages::-webkit-scrollbar-thumb:hover,
textarea::-webkit-scrollbar-thumb:hover {
  background: #b5b0a8;
}
</style>
