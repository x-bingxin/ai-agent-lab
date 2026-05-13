<script setup lang="ts">
import ts from 'typescript'
import {ref, nextTick} from 'vue'

interface Message {
  role: 'user' | 'assistant'
  content: string
  isStreaming?: boolean
}

const messages = ref<Message[]>([])
const inputText = ref('')
const chatContainer = ref<HTMLElement | null>(null)

async function sendMessage() {
  if (inputText.value.trim()) {

    // add user message
    messages.value.push({role: 'user', content: inputText.value})

    messages.value.push({role: 'assistant', content: '', isStreaming: true})
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
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({message: userMessage})
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
        const {value, done} = await reader.read()
        
        if (done) break

        buffer += decoder.decode(value, {stream: true})
        // console.log('Received chunk:', buffer)

        const lines = buffer.split('\n\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.substring(6)
            console.log('Parsed data:', data)
            if (data === '[DONE]') {
              updateAssistantMessage({isStreaming: false})
            } else if (data.startsWith('ERROR:')) {
              updateAssistantMessage({content: data, isStreaming: false})
            } else if (data !== 'None') {
              const current = messages.value[assistantIndex]
              updateAssistantMessage({content: current?.content + data})
              await nextTick()
              scrollToBottom()
            }
          }
        }
      }
    } catch (error) {
      console.error('Error:', error)
      updateAssistantMessage({content: `请求失败: ${error}`, isStreaming: false})
    } finally {
      updateAssistantMessage({isStreaming: false})
    }
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
  <div class="chat-container">
    <div class="messages" ref="chatContainer">
      <div
        v-for="(msg, idx) in messages"
        :key="idx"
        :class="['message', msg.role]">
        <div class="content">{{ msg.content }}</div>
        <span v-if="msg.isStreaming" class="cursor">|</span>
      </div>
    </div>
    <div class="input-area">
      <input
        v-model="inputText"
        @keyup.enter="sendMessage"
        placeholder="输入消息..."
      />
      <button @click="sendMessage">发送</button>
    </div>
  </div>
</template>

<style scoped>
.chat-container { 
  width: 1280px; 
  margin: 0 auto; 
}
.messages { 
  height: 500px; 
  overflow-y: auto; 
  border: 1px solid #ddd; 
  padding: 16px; 
}
.message { 
  margin-bottom: 12px; 
}
.message.user {
  background: #e3f2fd; 
  padding: 8px 12px; 
  border-radius: 8px; 
} 
.message.user .content { 
  color: #222;
  
}
.message.assistant {
  background: #f5f5f5; 
  padding: 8px 12px; 
  border-radius: 8px; 
} 
.message.assistant .content { 
  color: #555;  
}
.cursor { 
  color: #555;  
  animation: blink 1s infinite; 
}
@keyframes blink { 
  50% { opacity: 0; } 
}
.input-area { 
  display: flex; 
  margin-top: 12px; gap: 8px; 
}
.input-area input { 
  flex: 1; 
  padding: 8px; 
}
</style>
