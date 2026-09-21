/* Vantage RAG — embeddable chat widget.

   Paste on any page:
   <script src="https://api.kartikworks.co.in/widget/chat.js"
           data-api="https://api.kartikworks.co.in"
           data-title="Ask Kartik"></script>

   Optional data-* attributes:
     data-api        backend base URL (defaults to the widget's origin)
     data-title      chat header text
     data-subtitle   small line under the title
     data-accent     CSS color (e.g. #5bb8d8)
     data-position    left | right
     data-greeting   first bot message
*/
(function () {
  "use strict";

  var script = document.currentScript || Array.prototype.slice
    .call(document.getElementsByTagName("script")).pop();
  var api = script.getAttribute("data-api") || window.location.origin;
  var title = script.getAttribute("data-title") || "Chat with me";
  var subtitle = script.getAttribute("data-subtitle") || "Powered by Vantage RAG";
  var accent = script.getAttribute("data-accent") || "";
  var position = script.getAttribute("data-position") || "right";
  var greeting = script.getAttribute("data-greeting") ||
    "Hi! Ask me about skills, projects, experience, or anything on my portfolio.";

  if (document.getElementById("vw-root")) return; // already mounted

  var cssUrl = script.src.replace(/chat\.js$/, "chat.css");
  var link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = cssUrl;
  document.head.appendChild(link);

  var root = document.createElement("div");
  root.id = "vw-root";

  function esc(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  function renderMarkdown(text) {
    var lines = String(text || "").split(/\n+/).filter(Boolean);
    var out = [];
    var listOpen = false;
    lines.forEach(function (line) {
      var m = line.match(/^\s*[-*]\s+(.*)$/);
      if (m) {
        if (!listOpen) { out.push("<ul>"); listOpen = true; }
        out.push("<li>" + m[1] + "</li>");
        return;
      }
      if (listOpen) { out.push("</ul>"); listOpen = false; }
      var inline = esc(line)
        .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
      out.push("<p>" + inline + "</p>");
    });
    if (listOpen) out.push("</ul>");
    return out.join("");
  }

  root.innerHTML =
    '<button id="vw-launcher" type="button" aria-label="Open chat">' +
      '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/></svg>' +
      '<span class="vw-label">' + esc(title) + '</span>' +
    "</button>" +
    '<div id="vw-panel" class="vw-hide">' +
      '<div id="vw-head"><div><div class="vw-title">' + esc(title) + '</div>' +
      '<div class="vw-sub">' + esc(subtitle) + '</div></div>' +
      '<button id="vw-close" type="button" aria-label="Close">✕</button></div>' +
      '<div id="vw-msgs"></div>' +
      '<div id="vw-typing">Vantage is thinking</div>' +
      '<div id="vw-input-row">' +
        '<input id="vw-input" type="text" placeholder="Ask anything..." autocomplete="off" />' +
        '<button id="vw-send" type="button" aria-label="Send">' +
          '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>' +
        "</button>" +
      "</div>" +
    "</div>";
  document.body.appendChild(root);

  if (accent) root.style.setProperty("--vw-accent", accent);
  if (position === "left") {
    root.style.left = "22px"; root.style.right = "auto";
  }

  var launcher = root.querySelector("#vw-launcher");
  var panel = root.querySelector("#vw-panel");
  var msgs = root.querySelector("#vw-msgs");
  var typing = root.querySelector("#vw-typing");
  var input = root.querySelector("#vw-input");
  var send = root.querySelector("#vw-send");
  var close = root.querySelector("#vw-close");

  var threadKey = "vw_thread_" + window.location.hostname;
  var threadId = localStorage.getItem(threadKey) || null;

  function addMsg(text, cls) {
    var div = document.createElement("div");
    div.className = "vw-msg " + cls;
    div.innerHTML = cls === "bot" ? renderMarkdown(text) : esc(text);
    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;
    return div;
  }

  function setBusy(busy) {
    typing.style.display = busy ? "block" : "none";
    send.disabled = busy;
    input.disabled = busy;
  }

  function genThreadId() {
    if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
    return "t-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 10);
  }

  function handleEvent(part) {
    var text = String(part).trim();
    if (!text || text.indexOf("data:") !== 0) return;
    var data;
    try { data = JSON.parse(text.slice(5).trim()); } catch (e) { return; }

    if (data.type === "node" && data.status) {
      typing.textContent = "Vantage is thinking — " + data.status;
    } else if (data.type === "done") {
      setBusy(false);
      typing.textContent = "Vantage is thinking";
      addMsg(data.answer || "Hmm, I didn't get a response.", "bot");
    } else if (data.type === "error") {
      throw new Error(data.message || "Server error.");
    }
  }

  function ask(text) {
    addMsg(text, "user");
    input.value = "";
    if (!threadId) {
      threadId = genThreadId();
      localStorage.setItem(threadKey, threadId);
    }
    setBusy(true);
    typing.textContent = "Vantage is thinking";

    var payload = { q: text, thread_id: threadId };
    fetch(api.replace(/\/$/, "") + "/query/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then(function (resp) {
        if (!resp.ok) {
          return resp.json().then(function (body) {
            throw new Error((body && body.message) || ("HTTP " + resp.status));
          });
        }
        if (!resp.body) {
          return resp.json().then(function (data) {
            setBusy(false);
            addMsg(data.answer || "Hmm, I didn't get a response.", "bot");
          });
        }
        var reader = resp.body.getReader();
        var decoder = new TextDecoder("utf-8");
        var buffer = "";
        function pump(streamResult) {
          if (streamResult.done) {
            setBusy(false);
            return;
          }
          buffer += decoder.decode(streamResult.value, { stream: true });
          var parts = buffer.split("\n\n");
          buffer = parts.pop();
          parts.forEach(handleEvent);
          return reader.read().then(pump);
        }
        return reader.read().then(pump);
      })
      .catch(function (err) {
        setBusy(false);
        showError(err.message || "Something went wrong.");
      });
  }

  function showError(msg) {
    setBusy(false);
    var wrap = document.createElement("div");
    wrap.className = "vw-err";
    var span = document.createElement("span");
    span.textContent = "Backend hiccup — " + msg + ". Please retry in a moment.";
    var btn = document.createElement("button");
    btn.id = "vw-retry";
    btn.type = "button";
    btn.textContent = "Retry";
    btn.addEventListener("click", function () {
      wrap.remove();
      ask(input.value || "Hi there!");
    });
    wrap.appendChild(span);
    wrap.appendChild(btn);
    msgs.appendChild(wrap);
    msgs.scrollTop = msgs.scrollHeight;
  }

  function openChat() {
    launcher.classList.add("vw-hide");
    panel.classList.remove("vw-hide");
    if (!threadId) addMsg(greeting, "bot");
    input.focus();
  }
  function closeChat() {
    panel.classList.add("vw-hide");
    launcher.classList.remove("vw-hide");
  }

  launcher.addEventListener("click", openChat);
  close.addEventListener("click", closeChat);
  send.addEventListener("click", function () {
    var v = (input.value || "").trim();
    if (v) ask(v);
  });
  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter") {
      var v = (input.value || "").trim();
      if (v) ask(v);
    }
  });
})();