const ROUTES = ['chess_play', 'chess_status', 'browser_task', 'memory', 'chat'];

export async function routeMessage(message, settings, livePage = null) {
  const msg = String(message).trim();
  const lower = msg.toLowerCase();

  // Fast-path 1: Chess controls
  if (/\b(play|win|beat|move|your move)\b/.test(lower) && /\b(chess|game|match|opponent|board)\b/.test(lower)) {
    return { route: 'chess_play', via: 'fast-regex' };
  }
  if (/^\s*(status|game status|how's the game|whos winning|who's winning)/i.test(msg)) {
    return { route: 'chess_status', via: 'fast-regex' };
  }

  // Fast-path 2: Memory commands
  if (/\b(remember|don'?t forget|keep in mind|memorize)\b/i.test(lower)) {
    return { route: 'memory', via: 'fast-regex' };
  }

  // Fast-path 3: Explicit browser commands
  if (/^(go to|navigate to|open|browse to|search for|look up|click on|type into)\s+/i.test(msg) ||
      /\b(fill out|sign up on|scrape|read the active tab|inspect the page)\b/i.test(lower)) {
    return { route: 'browser_task', instruction: msg, via: 'fast-regex' };
  }

  // Fast-path 3b: Video / YouTube discovery — must be real browsing, not chat hallucination
  if (/\b(find|search|show|get|recommend|suggest|give)\b.{0,42}\bvideos?\b/i.test(lower) ||
      /\b(find|search|show|get|recommend|suggest|give)\b.{0,42}\byoutube\b/i.test(lower) ||
      /\b(similar|another|related)\b.{0,32}\bvideos?\b/i.test(lower) ||
      /\bmore (explained|detailed|beginner|simple)\b.{0,20}\bvideos?\b/i.test(lower) ||
      /related videos?/i.test(lower)) {
    return { route: 'browser_task', instruction: `Search YouTube for: ${msg} (use current video as context if present)`, via: 'video-search-regex' };
  }

  // Fast-path 4: CONTEXT-AWARE - action verb + reference to current tab/page/model/chat
  const actionVerb = /\b(message|prompt|type|click|read|check|fill|submit|post|send|interact|convince|negotiate|talk to|ask|jailbreak|probe|red.?team)\b/.test(lower);
  const deictic = /\b(this|that|the)\b.{0,25}\b(tab|page|model|chat|site|conversation|window|ai|bot|assistant|instance)\b/.test(lower) || /\bthis tab\b/i.test(msg);
  if (actionVerb && deictic && livePage && livePage.url && !livePage.url.startsWith('chrome')) {
    return {
      route: 'browser_task',
      instruction: `${msg} (Target: the page currently open in the linked browser - "${livePage.title}" at ${livePage.url})`,
      via: 'context-regex'
    };
  }

  // Fast-path 5: Standard questions/conversation -> chat (zero latency)
  if (lower.startsWith('explain') || lower.startsWith('what') || lower.startsWith('how') || lower.startsWith('why') ||
      lower.startsWith('write') || lower.startsWith('code') || lower.startsWith('debug') ||
      msg.length > 200 || msg.includes('?') || msg.includes('\n')) {
    return { route: 'chat', via: 'fast-heuristic' };
  }

  // Ambiguous short inputs: default chat (NO LLM routing - one NIM call per message max)
  return { route: 'chat', via: 'default' };
}
