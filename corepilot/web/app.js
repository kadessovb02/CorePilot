"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const activeStatuses = new Set(["queued", "running"]);
  const resumableStatuses = new Set(["cancelled", "failed", "interrupted"]);
  const statusNames = { queued: "Queued", running: "Running", completed: "Completed", cancelled: "Stopped", failed: "Error", interrupted: "Interrupted" };
  const state = { projects: [], tasks: new Map(), selected: null, project: "", model: null, modelMode: null, orchestration: null, controlled: null, resourceGuard: null, taskCapacity: {max_concurrent_tasks: 1, running: 0, queued: 0}, training: null, connected: false, busy: false, inspector: false, inspectorTab: "activity", lastTaskRefresh: 0, lastHealthRefresh: 0, rendered: "", activityRendered: "", diff: "", diffTask: null, pending: new Set(), polling: false };
  const handoff = { provider: "all", project: "", selected: null, chats: [], cache: new Map(), loading: false, busy: false, generation: 0, error: null };
  let toastTimer, pollTimer;

  function icon(name) { return `<svg aria-hidden="true"><use href="#i-${name}"/></svg>`; }
  function el(tag, className, text) { const node = document.createElement(tag); if (className) node.className = className; if (text !== undefined) node.textContent = text; return node; }
  function escape(value) { return String(value ?? "").replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char])); }
  function inline(text) { return escape(text).replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>"); }
  function formatMarkdown(value) {
    const fragment = document.createDocumentFragment();
    const parts = String(value ?? "").split(/(```[\s\S]*?(?:```|$))/g);
    for (const part of parts) {
      if (!part) continue;
      if (part.startsWith("```")) {
        const body = part.replace(/^```[^\n]*\n?/, "").replace(/```$/, "");
        const pre = el("pre"); pre.append(el("code", "", body.trimEnd())); fragment.append(pre); continue;
      }
      let paragraph = [], list = null;
      const flush = () => { if (paragraph.length) { const p = el("p"); p.innerHTML = inline(paragraph.join("\n")); fragment.append(p); paragraph = []; } list = null; };
      for (const line of part.split("\n")) {
        if (!line.trim()) { flush(); continue; }
        const heading = line.match(/^#{1,6}\s+(.+)/);
        if (heading) { flush(); const h = el("h3"); h.innerHTML = inline(heading[1]); fragment.append(h); continue; }
        const bullet = line.match(/^\s*(?:[-*]|\d+\.)\s+(.+)/);
        if (bullet) {
          if (paragraph.length) flush();
          if (!list) { list = el(/^\s*\d+\./.test(line) ? "ol" : "ul"); fragment.append(list); }
          const li = el("li"); li.innerHTML = inline(bullet[1]); list.append(li);
        } else { list = null; paragraph.push(line); }
      }
      flush();
    }
    return fragment;
  }
  function dateLabel(value, options = {}) { if (!value) return ""; const d = new Date(typeof value === "number" && value < 100000000000 ? value * 1000 : value); return Number.isNaN(d.getTime()) ? "" : new Intl.DateTimeFormat("en", options).format(d); }
  function clockLabel(value) { return dateLabel(value, { hour: "2-digit", minute: "2-digit" }); }
  function shortDate(value) { return dateLabel(value, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }); }
  function displayError(error) { return error?.message || String(error || "Unknown error"); }
  function savePreference(key, value) { try { value === null ? localStorage.removeItem(key) : localStorage.setItem(key, value); } catch { /* Browser storage is optional; backend persists the tasks. */ } }
  function showError(error) { $("form-error").textContent = displayError(error); $("form-error").hidden = false; }
  function clearError() { $("form-error").hidden = true; $("form-error").textContent = ""; }
  function toast(message) { clearTimeout(toastTimer); $("toast").textContent = message; $("toast").hidden = false; toastTimer = setTimeout(() => { $("toast").hidden = true; }, 5000); }
  function currentTask() { return state.tasks.get(state.selected); }
  function runningTask() { return [...state.tasks.values()].find((task) => activeStatuses.has(task.status)); }
  function blocksAnotherTask() { return !!runningTask() && !(state.taskCapacity?.max_concurrent_tasks > 1); }
  function detailString(data) { if (data === null || data === undefined) return ""; if (typeof data === "string") return data; try { return JSON.stringify(data, null, 2); } catch { return String(data); } }
  function eventText(event) { return event.message || event.data?.text || event.data?.content || ""; }
  function toolName(event) { return event.data?.tool || event.data?.name || event.data?.tool_name || event.message || "Tool"; }
  function eventData(event) { const d = event.data; if (!d) return ""; return detailString(d.result ?? d.output ?? d.arguments ?? d.args ?? d.input ?? d); }
  function providerName(provider) { return ({codex: "Codex", claude: "Claude", canonical: "JSONL"})[provider] || "History"; }
  function roleName(role) { return ({planner: "Planner", coder: "Coder", reviewer: "Reviewer", select: "Inspect", edit: "Edit", tests: "Tests", check: "Verify", review: "Review", inspect: "Explain"})[role] || role || "Agent"; }
  function workflowChoice() { return $("workflow-select").value === "single" ? false : $("workflow-select").value === "specialists" ? true : null; }
  function useControlled(task = null) { return task ? !!task.controlled?.enabled : $("workflow-select").value === "controlled" || ($("workflow-select").value === "auto" && !!state.controlled?.enabled); }
  function useSpecialists(task = null) { return task ? !!task.orchestration?.enabled : !useControlled() && (workflowChoice() ?? !!state.orchestration?.enabled); }
  function modelReady(task = null) { return state.connected && !state.resourceGuard?.blocked && (useSpecialists(task) ? !!state.orchestration?.ready : state.modelMode !== "specialists" && !!state.model?.ready); }
  function addWorkflow(payload) { const mode = $("workflow-select").value; if (mode === "controlled") { payload.controlled = true; payload.orchestrated = false; } else if (mode !== "auto") { payload.controlled = false; payload.orchestrated = mode === "specialists"; } return payload; }
  function projectBranch(project) { return project?.branch || (project?.head ? String(project.head).slice(0, 7) : ""); }
  function projectDescription(project) { return [project?.path, project?.branch ? `Branch: ${project.branch}` : project?.head ? "State: detached HEAD" : null, project?.head ? `Commit: ${project.head}` : null, project?.dirty ? "Uncommitted changes" : null].filter(Boolean).join("\n"); }

  async function api(path, options = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), options.timeout || 20000);
    try {
      const response = await fetch(path, { ...options, headers: { "Content-Type": "application/json", ...options.headers }, signal: controller.signal });
      const text = await response.text();
      let body; try { body = text ? JSON.parse(text) : {}; } catch { throw new Error(`The server returned an invalid response (${response.status}).`); }
      if (!response.ok) { const detail = body.error || body.detail || body.message; throw new Error(typeof detail === "string" ? detail : detailString(detail) || `Server error: ${response.status}`); }
      return body;
    } catch (error) {
      if (error.name === "AbortError") throw new Error("The server is taking too long. Check the local process and try again.");
      if (error instanceof TypeError) throw new Error("Cannot connect to the local server. Run corepilot serve and refresh this page.");
      throw error;
    } finally { clearTimeout(timer); }
  }

  function mergeTask(snapshot) {
    if (!snapshot?.id) return;
    const prior = state.tasks.get(snapshot.id);
    const events = new Map((prior?.events || []).map((event) => [event.seq, event]));
    for (const event of snapshot.events || []) events.set(event.seq, event);
    state.tasks.set(snapshot.id, { ...prior, ...snapshot, events: [...events.values()].sort((a, b) => a.seq - b.seq) });
  }

  function setProject(id) {
    const available = state.projects.find((project) => project.id === id && project.available !== false);
    if (!available) return;
    state.project = id;
    $("project-select").value = id;
    savePreference("corepilot-project", id);
    renderProjects(); updateComposer();
  }

  function renderProjects() {
    const selectedProject = currentTask()?.project_id || currentTask()?.project || state.project;
    $("project-list").replaceChildren();
    $("project-select").replaceChildren();
    for (const project of state.projects) {
      const option = el("option", "", project.name || project.id); option.value = project.id; option.disabled = project.available === false; $("project-select").append(option);
      const button = el("button", "project-item" + (selectedProject === project.id ? " selected" : ""));
      button.type = "button"; button.disabled = project.available === false;
      button.innerHTML = icon("branch");
      const text = el("span", "project-item-text"); text.append(el("span", "project-item-name", project.name || project.id));
      const branch = projectBranch(project);
      if (branch) text.append(el("span", "project-item-branch", branch + (project.dirty ? " · modified" : "")));
      button.append(text, el("span", "project-ready"));
      button.title = projectDescription(project) || project.id;
      button.setAttribute("aria-label", (project.name || project.id) + (branch ? `, branch ${branch}` : ""));
      button.addEventListener("click", () => { if (state.selected) newTask(); setProject(project.id); closeSidebar(); });
      $("project-list").append(button);
    }
    if (!state.projects.length) { $("project-list").append(el("span", "muted loading-label", "No projects configured")); const option = el("option", "", "No available projects"); option.value = ""; $("project-select").append(option); }
    $("project-select").value = selectedProject;
    let project = state.projects.find((item) => item.id === selectedProject);
    const task = currentTask();
    if (task && ("source_branch" in task || "source_head" in task)) project = { ...project, branch: task.source_branch, head: task.source_head, dirty: task.source_dirty };
    const branch = projectBranch(project);
    $("project-branch").hidden = !branch;
    $("project-branch").textContent = branch;
    $("project-branch").title = projectDescription(project);
    $("project-branch").setAttribute("aria-label", branch ? `Current project branch: ${branch}` : "");
    const available = state.projects.some(project => project.available !== false);
    $("setup-panel").hidden = available || !state.connected;
    $("welcome").classList.toggle("setup-welcome", !available && state.connected);
    $("welcome-title").textContent = !available && state.connected ? "Connect your first project" : "What shall we work on?";
    $("welcome-description").textContent = !available && state.connected ? "A single setup command connects your repository and local model." : "Describe a task. Your agent reads code, makes changes, and runs checks in an isolated workspace.";
    $("import-history").disabled = !available;
    $("welcome-import").hidden = !available;
    document.querySelector(".suggestions").hidden = !available && state.connected;

  }

  function renderTaskList() {
    const tasks = [...state.tasks.values()].sort((a, b) => (b.created_at || "").localeCompare(a.created_at || ""));
    $("history-count").textContent = tasks.length ? String(tasks.length) : "";
    const nodes = tasks.map((task) => {
      const button = el("button", "task-item" + (state.selected === task.id ? " selected" : "")); button.type = "button";
      button.append(el("span", `task-indicator ${task.status || ""}`));
      const text = el("span", "task-item-text"); text.append(el("span", "task-item-title", task.prompt || "Task"), el("span", "task-item-meta", `${task.project_id || task.project || ""} · ${shortDate(task.created_at)}`)); button.append(text); button.title = task.prompt || "Task";
      button.setAttribute("aria-current", state.selected === task.id ? "true" : "false");
      button.addEventListener("click", () => selectTask(task.id)); return button;
    });
    if (!nodes.length) nodes.push(el("div", "sidebar-empty", "Saved tasks will appear here"));
    $("task-list").replaceChildren(...nodes);
  }

  function renderModel() {
    const model = state.model;
    const task = currentTask();
    const pinned = task && (activeStatuses.has(task.status) || resumableStatuses.has(task.status)) ? task : null;
    const ready = modelReady(pinned);
    $("resource-hold-panel").hidden = !state.resourceGuard?.blocked;
    $("model-dot").className = "status-dot " + (ready ? "ready" : "error");
    const name = (model?.display_name || model?.name)?.split("/").pop();
    $("model-label").textContent = state.connected ? (useSpecialists(pinned) ? "Specialists" : name || (model?.ready ? "Model ready" : "Model offline")) : "Disconnected";
    $("model-button").title = ready ? "Configured workflow models are ready" : "Show model status";
    const details = [["Status", state.connected ? (model?.ready ? "Ready" : "Unavailable") : "Server unavailable"], ["Model", model?.display_name || model?.name || "—"], ["Adapter", model?.adapter || "None"], ["Model endpoint", model?.url || "—"]];
    if (model?.error) details.push(["Reason", typeof model.error === "string" ? model.error : detailString(model.error)]);
    $("model-details").replaceChildren(...details.flatMap(([label, value]) => [el("dt", "", label), el("dd", "", value)]));
    renderRoleModels();
    renderTraining();
    if (state.connected) {
      $("connection-banner").hidden = ready;
      if (!ready) $("connection-message").textContent = state.resourceGuard?.blocked ? "Computation paused by the resource guard. Results are saved. Stop the model server and review the resource hold before resuming." : useSpecialists(pinned) ? "Specialist models unavailable. Check the configured role endpoints; saved tasks remain accessible." : "Model unavailable. Configure a running local endpoint; saved tasks and diffs remain accessible.";
    }
    updateComposer();
  }

  function renderRoleModels() {
    const pool = state.orchestration;
    const configured = !!pool && (pool.configured ?? Object.keys(pool.roles || {}).length > 0);
    const anyRoles = configured || Object.values(pool?.roles || {}).some(role => role.configured);
    $("workflow-option").hidden = !configured && !state.controlled?.configured;
    if (!configured && !state.controlled?.configured) $("workflow-select").value = "auto";
    $("workflow-select").options[0].textContent = `Auto · ${state.controlled?.enabled ? "controlled" : pool?.enabled ? "specialists" : "single"}`;
    $("role-model-details").hidden = !anyRoles;
    if (!anyRoles) return;
    const nodes = [el("div", "role-model-heading", "Planner → Coder → Reviewer")];
    for (const role of ["planner", "coder", "reviewer"]) {
      const model = pool.roles?.[role];
      const row = el("div", "role-model-row");
      row.append(el("span", "status-dot " + (model?.ready ? "ready" : "error")), el("span", "", roleName(role)));
      row.append(el("code", "", model?.display_name || model?.name || "Unconfigured"));
      row.title = model?.ready ? "Model ready" : model?.error || "Model unavailable";
      nodes.push(row);
    }
    nodes.push(el("p", "", "Roles run in order using your configured model servers."));
    $("role-model-details").replaceChildren(...nodes);
  }

  function renderWorkflow(task) {
    const workflow = task?.controlled?.enabled ? task.controlled : task?.orchestration;
    $("workflow-status").hidden = !workflow?.enabled;
    if (!workflow?.enabled) return;
    const nodes = [];
    for (const role of task?.controlled?.enabled ? ["select", "edit", "tests", "check", "review"] : ["planner", "coder", "reviewer"]) nodes.push(el("span", "workflow-phase" + (workflow.phase === role ? " current" : ""), roleName(role)));
    const round = typeof workflow.round === "number" && workflow.round > 1 ? ` · revision ${workflow.round - 1}` : "";
    nodes.push(el("small", "", `${activeStatuses.has(task.status) ? "Working" : statusNames[task.status] || task.status}${round}`));
    $("workflow-status").replaceChildren(...nodes);
  }

  function renderTraining() {
    const training = state.training;
    $("training-status").hidden = !training || training.status === "unavailable";
    if (!training || training.status === "unavailable") return;
    const labels = { planned: "Training configured", running: "Adapter training in progress", complete: "Training complete", completed: "Training complete", failed: "Training failed" };
    const header = el("div", `training-status-header ${training.status || ""}`);
    header.append(el("span", "status-dot " + (training.status === "failed" ? "error" : ["complete", "completed"].includes(training.status) ? "ready" : "")), el("span", "", labels[training.status] || "Adapter training"));
    const nodes = [header];
    const parts = [];
    if (typeof training.training_rows === "number") parts.push(`${training.training_rows.toLocaleString("en")} training examples`);
    if (typeof training.validation_rows === "number") parts.push(`${training.validation_rows.toLocaleString("en")} validation examples`);
    if (parts.length) nodes.push(el("p", "", parts.join(" · ")));
    const progress = training.progress;
    if (typeof progress?.iteration === "number") nodes.push(el("p", "training-iteration", `Iteration ${progress.iteration.toLocaleString("en")}`));
    if (typeof training.elapsed_seconds === "number") nodes.push(el("p", "training-time", `Elapsed: ${Math.round(training.elapsed_seconds / 60)} min`));
    if (training.error) nodes.push(el("p", "training-error", String(training.error)));
    $("training-status").replaceChildren(...nodes);
  }

  function showConnectionError(error) {
    state.connected = false;
    $("connection-message").textContent = displayError(error);
    $("connection-banner").hidden = false;
    renderModel();
  }

  async function refreshHealth() {
    if (state.pending.has("health")) return;
    state.pending.add("health");
    try {
      const task = currentTask();
      const pinned = task && (activeStatuses.has(task.status) || resumableStatuses.has(task.status)) ? task : null;
      const choice = pinned ? !!pinned.orchestration?.enabled : useControlled() ? false : workflowChoice();
      const status = await api("/api/status" + (choice === null ? "" : `?workflow=${choice ? "specialists" : "single"}`));
      state.connected = true; state.model = status.model || status; state.modelMode = status.model_mode || null; state.orchestration = status.orchestration || null; state.controlled = status.controlled || null; state.resourceGuard = status.resource_guard || null; state.taskCapacity = status.task_capacity || {max_concurrent_tasks: 1, running: 0, queued: 0}; state.training = status.training || null;
      if (Array.isArray(status.projects)) {
        state.projects = status.projects;
        if (!state.project || !state.projects.find((project) => project.id === state.project && project.available !== false)) state.project = state.projects.find((project) => project.available !== false)?.id || "";
        renderProjects();
      }
      state.lastHealthRefresh = Date.now();
      renderModel();
    } catch (error) { showConnectionError(error); }
    finally { state.pending.delete("health"); }
  }

  async function refreshTasks() {
    if (state.pending.has("tasks")) return;
    state.pending.add("tasks");
    try {
      const result = await api("/api/tasks");
      const tasks = Array.isArray(result) ? result : result.tasks || [];
      for (const task of tasks) mergeTask(task);
      state.lastTaskRefresh = Date.now(); renderTaskList(); updateComposer();
    } catch (error) { if (state.connected) showConnectionError(error); }
    finally { state.pending.delete("tasks"); }
  }

  function newTask() {
    state.selected = null; state.rendered = ""; state.activityRendered = ""; state.diff = ""; state.diffTask = null;
    savePreference("corepilot-task", null);
    $("page-name").textContent = "New task";
    $("task-status").hidden = true; $("welcome").hidden = false; $("messages").hidden = true; $("messages").replaceChildren();
    $("checkpoint-card").hidden = true; renderEmptyActivity(); renderEmptyDiff();
    renderWorkflow(null);
    $("workflow-select").value = "auto";
    $("prompt").value = ""; resizePrompt(); clearError(); renderProjects(); renderTaskList(); updateComposer();
    state.lastHealthRefresh = 0; renderModel(); refreshHealth();
    closeSidebar(); $("prompt").focus();
  }

  async function selectTask(id) {
    if (!state.tasks.has(id)) return;
    state.selected = id; state.rendered = ""; state.activityRendered = ""; state.diff = ""; state.diffTask = null;
    savePreference("corepilot-task", id);
    clearError(); closeSidebar();
    renderProjects(); renderTaskList(); renderTask(); renderEmptyDiff();
    await refreshTask(true);
    await refreshHealth();
    if (state.selected === id && state.inspectorTab === "diff") await refreshDiff();
  }

  async function refreshTask(full = false) {
    const id = state.selected;
    if (!id || state.pending.has(id)) return;
    state.pending.add(id);
    try {
      const old = state.tasks.get(id);
      const after = full ? 0 : (old?.events?.at(-1)?.seq || 0);
      const task = await api(`/api/tasks/${encodeURIComponent(id)}?after=${after}`);
      const oldStatus = old?.status;
      mergeTask(task);
      if (state.selected === id) { renderTask(); if (oldStatus !== task.status && !activeStatuses.has(task.status)) { renderTaskList(); if (state.inspector && state.inspectorTab === "diff") await refreshDiff(); } }
    } catch (error) { if (state.selected === id) showError(error); }
    finally { state.pending.delete(id); }
  }

  function makeAssistant(text, time) {
    const message = el("div", "message assistant-message");
    const header = el("div", "message-header"); const avatar = el("span", "avatar"); avatar.innerHTML = icon("terminal");
    header.append(avatar, el("span", "", "CorePilot"), el("small", "", clockLabel(time)));
    const body = el("div", "markdown"); body.append(formatMarkdown(text)); message.append(header, body); return message;
  }

  function makeTool(event, result) {
    const output = result?.data?.result ?? (event.type === "tool_result" ? event.data?.result : null);
    const failed = !!(output?.error || output?.timed_out || (typeof output?.exit_code === "number" && output.exit_code !== 0));
    const details = el("details", "tool-chip" + (failed ? " tool-failed" : "")); details.dataset.eventSeq = event.seq;
    const summary = el("summary"); summary.innerHTML = icon(failed ? "close" : result || event.type === "tool_result" ? "check" : "terminal");
    const label = (event.data?.role ? `${roleName(event.data.role)} · ` : "") + toolName(event) + (failed ? (output.exit_code !== undefined ? ` · exit ${output.exit_code}` : " · error") : "");
    summary.append(el("span", "tool-name", label), el("span", "tool-time", clockLabel(result?.time || event.time)));
    const chevron = document.createElementNS("http://www.w3.org/2000/svg", "svg"); chevron.classList.add("tool-chevron"); chevron.innerHTML = '<use href="#i-chevron"/>'; summary.append(chevron); details.append(summary);
    const text = result ? `Call\n${eventData(event)}\n\nResult\n${eventData(result)}` : eventData(event);
    details.append(el("pre", "", text ? text.slice(0, 32000) : eventText(event))); return details;
  }

  function taskChain(task) {
    const chain = [], seen = new Set();
    for (let next = task; next && !seen.has(next.id); next = next.parent_id ? state.tasks.get(next.parent_id) : null) { seen.add(next.id); chain.unshift(next); }
    return chain;
  }

  function renderTask() {
    const task = currentTask(); if (!task) return;
    $("page-name").textContent = task.prompt?.trim().split("\n")[0] || "Task";
    $("task-status").textContent = statusNames[task.status] || task.status;
    $("task-status").className = `status-pill ${task.status}`; $("task-status").hidden = false;
    $("welcome").hidden = true; $("messages").hidden = false;
    renderWorkflow(task);
    const chain = taskChain(task);
    const signature = JSON.stringify(chain.map((item) => [item.id, item.status, item.answer, item.error, item.events?.length, item.events?.at(-1)?.seq, item.updated_at, item.orchestration, item.controlled]));
    if (state.rendered !== signature) {
      const scroll = $("chat-scroll"); const atBottom = scroll.scrollHeight - scroll.clientHeight - scroll.scrollTop < 120;
      const opened = new Set([...$("messages").querySelectorAll("details[open]")].map((item) => item.dataset.eventSeq));
      const nodes = [];
      for (const item of chain) {
        nodes.push(el("div", "user-message", item.prompt || ""));
        if (item.history_session_id || item.codex_session_id) {
          const note = el("div", "handoff-note" + (item.import_mode === "configured-fallback" ? " fallback" : ""));
          note.append(el("strong", "", `Context imported from ${providerName(item.history_provider || "codex")}`), el("span", "", item.source_note || "This task continues from a session summary using the project’s current files."));
          nodes.push(note);
        }
        let lastAssistant = "";
        const toolResults = new Map((item.events || []).filter((event) => event.type === "tool_result" && event.data?.call_id).map((event) => [event.data.call_id, event]));
        const pairedCalls = new Set((item.events || []).filter((event) => event.type === "tool_start" && event.data?.call_id).map((event) => event.data.call_id));
        for (const event of item.events || []) {
          if (event.type === "role") { const role = el("div", "role-transition"); role.append(el("strong", "", roleName(event.data?.role)), el("span", "", event.data?.model || "")); if (event.data?.round > 1) role.append(el("small", "", `Revision ${event.data.round - 1}`)); nodes.push(role); }
          if (event.type === "user") nodes.push(el("div", "user-message", eventText(event)));
          if (event.type === "assistant" && eventText(event)) { lastAssistant = eventText(event); nodes.push(makeAssistant(lastAssistant, event.time)); }
          if (event.type === "tool_start" || (event.type === "tool_result" && !pairedCalls.has(event.data?.call_id))) { const tool = makeTool(event, toolResults.get(event.data?.call_id)); if (opened.has(String(event.seq))) tool.open = true; nodes.push(tool); }
        }
        if (item.answer && item.answer.trim() !== lastAssistant.trim()) nodes.push(makeAssistant(item.answer, item.updated_at));
        const review = item.controlled?.review || item.orchestration?.review;
        if (review) { const note = el("details", "workflow-review" + (review.approved ? " approved" : "")); const summary = el("summary", "", review.approved ? "Reviewer approved the implementation" : "Reviewer requested changes"); note.append(summary); if (review.summary) note.append(el("p", "", review.summary)); if (review.issues?.length) { const issues = el("ul"); for (const issue of review.issues) issues.append(el("li", "", issue)); note.append(issues); } nodes.push(note); }
      }
      if (activeStatuses.has(task.status)) { const working = el("div", "working-row"); const dots = el("span", "thinking-dots"); dots.innerHTML = "<i></i><i></i><i></i>"; working.append(dots, el("span", "", task.status === "queued" ? "Task queued…" : `${task.controlled?.enabled ? roleName(task.controlled.phase) : task.orchestration?.enabled ? roleName(task.orchestration.phase) : "Agent"} working · step ${task.step || 0}`)); nodes.push(working); }
      else { const end = el("div", `task-end ${task.status}`); end.innerHTML = icon(task.status === "completed" ? "check" : "stop"); end.append(el("span", "", task.error || (task.status === "completed" ? "Task completed. Review the result and file changes." : task.status === "failed" ? "Task failed. You can resume from the saved checkpoint." : "Task stopped. Context is saved."))); nodes.push(end); }
      $("messages").replaceChildren(...nodes); state.rendered = signature;
      if (atBottom) requestAnimationFrame(() => { scroll.scrollTop = scroll.scrollHeight; });
    }
    renderActivity(task); updateComposer();
  }

  function renderEmptyActivity() {
    $("activity-list").innerHTML = '<div class="inspector-empty">' + icon("terminal") + '<span>Activity will appear here</span><p>File reads, commands, and check results appear when you start a task.</p></div>';
  }

  function renderActivity(task) {
    $("checkpoint-card").hidden = false;
    $("step-label").textContent = `Step ${task.step || 0} of ${task.max_steps || 24}`;
    $("step-status").textContent = task.controlled?.enabled && task.controlled.phase !== "completed" ? roleName(task.controlled.phase) : task.orchestration?.enabled && task.orchestration.phase !== "completed" ? roleName(task.orchestration.phase) : statusNames[task.status] || task.status;
    $("step-progress").style.width = `${Math.min(100, 100 * (task.step || 0) / (task.max_steps || 24))}%`;
    $("workspace-path").textContent = task.workspace || "Preparing workspace…";
    $("source-path").textContent = task.source_path || "—";
    $("source-branch").hidden = !task.source_branch && !task.source_head;
    $("source-branch").textContent = [task.source_branch ? `Branch: ${task.source_branch}` : task.source_head ? "detached HEAD" : "", task.source_head?.slice(0, 7)].filter(Boolean).join(" · ");
    const signature = `${task.id}:${task.events?.length}:${task.events?.at(-1)?.seq}`;
    if (signature === state.activityRendered) return;
    const list = $("activity-list"); const opened = new Set([...list.querySelectorAll("details[open]")].map((item) => item.dataset.eventSeq));
    const nodes = [];
    const labels = { status: "Status", role: "Role switch", routing: "Workflow choice", assistant: "Model response", user: "Task clarification", memory: "Project memory", model: "Model step", handoff: "History handoff", source: "Source workspace", tool_start: "Tool call", tool_result: "Tool result", error: "Error", diff: "File changes", controlled: "Workflow stage", resources: "Resource pause" };
    for (const event of task.events || []) {
      const item = el("div", `activity-item ${event.type}`); const header = el("div", "activity-header"); header.append(el("span", "activity-dot"), el("span", "", event.type.startsWith("tool_") ? toolName(event) : labels[event.type] || event.type)); const time = el("time", "", clockLabel(event.time)); time.dateTime = event.time || ""; header.append(time); item.append(header);
      if (event.type === "status") item.append(el("div", "activity-details", statusNames[event.data?.status] || eventText(event)));
      else if (event.type === "role") item.append(el("div", "activity-details", [roleName(event.data?.role), event.data?.model, event.data?.round > 1 ? `Revision ${event.data.round - 1}` : ""].filter(Boolean).join(" · ")));
      else if (event.type === "error") item.append(el("div", "activity-details", eventText(event)));
      else { const details = el("details"); details.dataset.eventSeq = event.seq; if (opened.has(String(event.seq))) details.open = true; details.append(el("summary", "", event.type === "assistant" ? "Show response" : "Show details"), el("pre", "", (eventData(event) || eventText(event)).slice(0, 32000))); item.append(details); }
      nodes.push(item);
    }
    if (!nodes.length) renderEmptyActivity(); else list.replaceChildren(...nodes);
    state.activityRendered = signature;
  }

  function renderEmptyDiff() {
    $("diff-count").textContent = ""; $("diff-summary").textContent = "No changes"; $("changed-files").replaceChildren();
    $("diff-content").innerHTML = '<div class="inspector-empty">' + icon("file") + '<span>File changes</span><p>Review the diff before applying changes to your source repository.</p></div>';
    $("copy-diff").disabled = true; $("apply-area").hidden = true;
  }

  async function refreshDiff() {
    const id = state.selected; if (!id || state.pending.has("diff")) return;
    state.pending.add("diff");
    try {
      const result = await api(`/api/tasks/${encodeURIComponent(id)}/diff`);
      if (state.selected !== id) return;
      state.diff = result.diff || ""; state.diffTask = id;
      const files = result.changed_files || [];
      $("diff-count").textContent = files.length ? String(files.length) : "";
      if (!state.diff && !files.length) { renderEmptyDiff(); return; }
      const additions = state.diff.split("\n").filter((line) => line.startsWith("+") && !line.startsWith("+++")).length;
      const deletions = state.diff.split("\n").filter((line) => line.startsWith("-") && !line.startsWith("---")).length;
      $("diff-summary").textContent = `${files.length} ${files.length === 1 ? "file" : "files"} · +${additions} −${deletions}`;
      $("changed-files").replaceChildren(...files.map((file) => { const row = el("div", "changed-file"); row.innerHTML = icon("file"); row.append(el("span", "", typeof file === "string" ? file : file.path || file.name || detailString(file))); return row; }));
      const pre = el("pre");
      const lines = state.diff.split("\n");
      for (const line of lines.slice(0, 2500)) {
        const type = line.startsWith("diff ") || line.startsWith("---") || line.startsWith("+++") ? "file" : line.startsWith("+") ? "addition" : line.startsWith("-") ? "deletion" : line.startsWith("@@") ? "hunk" : "";
        pre.append(el("span", `diff-line ${type}`, line || " "));
      }
      if (lines.length > 2500) pre.append(el("span", "diff-line hunk", `Showing the first 2500 of ${lines.length} lines. Copy saves the full diff.`));
      if (!state.diff) pre.append(el("span", "diff-line", "Binary files changed. Their content is not shown in the text diff."));
      $("diff-content").replaceChildren(pre); $("copy-diff").disabled = !state.diff;
      const task = currentTask(); $("apply-area").hidden = activeStatuses.has(task.status);
      $("apply-changes").disabled = !!task.applied;
      $("apply-changes").innerHTML = icon("check") + (task.applied ? "Changes applied" : "Apply to project");
      $("apply-description").textContent = task.applied ? "Changes were applied to the source repository." : "Copy these changes from the isolated workspace to your source repository.";
    } catch (error) { if (state.selected === id) { $("diff-content").replaceChildren(el("div", "inspector-empty", displayError(error))); } }
    finally { state.pending.delete("diff"); }
  }

  function updateComposer() {
    const task = currentTask(), running = runningTask(), active = task && activeStatuses.has(task.status), otherActive = running && running.id !== state.selected;
    const otherBlocked = otherActive && blocksAnotherTask();
    const pinned = task && resumableStatuses.has(task.status) ? task : null;
    $("send-button").hidden = !!active; $("stop-button").hidden = !active;
    $("stop-button").disabled = state.busy;
    $("send-button").disabled = state.busy || !modelReady(pinned) || !state.project || !$("prompt").value.trim() || !!otherBlocked;
    $("prompt").disabled = state.busy || !!active;
    $("prompt").placeholder = active ? "The agent is working. Stop the task to resume it later." : resumableStatuses.has(task?.status) ? "Add instructions to resume from the checkpoint…" : task ? "Next task for this project…" : "Describe what to do…";
    $("send-button").setAttribute("aria-label", resumableStatuses.has(task?.status) ? "Resume task with a message" : "Send task");
    $("project-select").disabled = state.busy || !!task || !state.projects.some((project) => project.available !== false);
    $("memory-toggle").disabled = state.busy || !!active;
    $("workflow-select").disabled = state.busy || !!active || !!pinned;
    if (active || pinned) $("workflow-select").value = task.controlled?.enabled ? "controlled" : task.orchestration?.enabled ? "specialists" : "single";
    $("task-actions").hidden = !task || !resumableStatuses.has(task.status);
    $("resume-task").disabled = state.busy || !modelReady(task) || blocksAnotherTask();
    $("checkpoint-label").textContent = task ? `Context saved · ${task.step || 0} steps` : "";
    $("composer-hint").textContent = !state.project && state.connected ? "Add a project to start a task" : otherBlocked ? "Another task is running" : active ? "You can stop the task without losing its checkpoint" : otherActive ? `${state.taskCapacity.running || 0} running · ${state.taskCapacity.queued || 0} queued` : "Enter to send · Shift + Enter for a new line";
    if ($("history-dialog").open) updateHandoffControls();
  }

  function updateHandoffControls() {
    $("continue-history").disabled = handoff.busy || handoff.loading || !handoff.selected || !modelReady() || blocksAnotherTask();
    $("close-history").disabled = handoff.busy;
    $("history-project").disabled = handoff.busy;
    $("history-provider").disabled = handoff.busy;
    $("history-prompt").disabled = handoff.busy;
    $("refresh-history").disabled = handoff.busy || handoff.loading;
    for (const button of $("history-chat-list").querySelectorAll(".history-chat-option")) button.disabled = handoff.busy;
    $("history-model-hint").textContent = handoff.busy ? "Importing context and creating a task…" : !modelReady() ? "Browse history now. Continue when the workflow models are ready." : blocksAnotherTask() ? "Finish or stop the active task first." : handoff.selected ? "The agent starts from the current files." : "Choose a session to continue.";
  }

  function renderHandoffList() {
    const list = $("history-chat-list");
    list.setAttribute("aria-busy", String(handoff.loading));
    if (handoff.loading) { list.replaceChildren(el("div", "history-list-empty", "Reading configured history sources…")); updateHandoffControls(); return; }
    if (!handoff.chats.length) { const empty = el("div", "history-list-empty"); empty.innerHTML = icon("file"); empty.append(el("span", "", handoff.error ? "History could not be read. Check the connection and click Refresh." : "No sessions found. Add a history source with corepilot history add, or select another project.")); list.replaceChildren(empty); updateHandoffControls(); return; }
    list.replaceChildren(...handoff.chats.map((chat) => {
      const button = el("button", "history-chat-option" + (chat.id === handoff.selected ? " selected" : ""));
      button.dataset.chatId = chat.id;
      button.dataset.importMode = chat.import_mode || "";
      button.type = "button"; button.setAttribute("role", "option"); button.setAttribute("aria-selected", String(chat.id === handoff.selected));
      button.append(el("span", "history-chat-title", chat.title || "Untitled session"));
      const metadata = el("span", "history-chat-meta"); metadata.append(el("span", "", `${providerName(chat.provider || "codex")} · ${shortDate(chat.updated_at || chat.date)}`), el("span", "", chat.source_branch ? `Branch: ${chat.source_branch}` : chat.branch ? `Branch: ${chat.branch}` : chat.source_exists ? "Workspace available" : "Workspace unavailable")); button.append(metadata);
      if (chat.source_cwd) button.append(el("code", "history-chat-path", chat.source_cwd));
      const note = chat.status_note || chat.statusnote;
      if (note) button.append(el("span", "history-chat-note" + (chat.import_mode === "configured-fallback" || chat.source_exists === false ? " fallback" : ""), note));
      button.addEventListener("click", () => { handoff.selected = chat.id; renderHandoffList(); updateHandoffNote(); });
      return button;
    }));
    updateHandoffControls();
  }

  function updateHandoffNote() {
    const chat = handoff.chats.find((item) => item.id === handoff.selected);
    const note = chat?.status_note || chat?.statusnote;
    $("history-import-note").textContent = note ? `${note} The summary becomes context for a new task. Old commands are not replayed.` : "A session summary becomes context for a new task. Old commands are not replayed. The current branch is checked at startup.";
  }

  async function loadHistoryChats(force = false) {
    const project = handoff.project; if (!project) return;
    const generation = ++handoff.generation;
    handoff.selected = null; handoff.chats = []; handoff.error = null; $("history-error").hidden = true; updateHandoffNote();
    const cacheKey = `${project}:${handoff.provider}`;
    const cached = handoff.cache.get(cacheKey);
    if (!force && cached && Date.now() - cached.fetchedAt < 300000) { handoff.chats = cached.chats; handoff.loading = false; renderHandoffList(); return; }
    handoff.loading = true; renderHandoffList();
    try {
      const result = await api(`/api/history/chats?project=${encodeURIComponent(project)}&provider=${encodeURIComponent(handoff.provider)}`);
      const chats = (Array.isArray(result.chats) ? result.chats : []).slice(0, 20);
      handoff.cache.set(cacheKey, { chats, fetchedAt: Date.now() });
      if (generation !== handoff.generation) return;
      handoff.chats = chats;
    } catch (error) {
      if (generation !== handoff.generation) return;
      handoff.error = displayError(error); $("history-error").textContent = handoff.error; $("history-error").hidden = false;
    } finally { if (generation === handoff.generation) { handoff.loading = false; renderHandoffList(); } }
  }

  function openHistoryDialog() {
    if (state.busy || !state.projects.some(project => project.available !== false)) { if (!state.projects.some(project => project.available !== false)) showError("No project available. Add a repository with corepilot project add."); return; }
    handoff.project = currentTask()?.project_id || currentTask()?.project || state.project;
    $("history-project").replaceChildren(...state.projects.filter((project) => project.available !== false).map((project) => { const option = el("option", "", `${project.name || project.id}${projectBranch(project) ? ` · ${projectBranch(project)}` : ""}`); option.value = project.id; return option; }));
    $("history-project").value = handoff.project;
    $("history-provider").value = handoff.provider;
    $("history-prompt").value = $("prompt").value.trim();
    closeSidebar(); $("history-dialog").showModal(); loadHistoryChats();
  }

  async function continueHistoryChat() {
    if ($("continue-history").disabled || !handoff.selected) return;
    handoff.busy = true; state.busy = true; $("history-error").hidden = true; updateComposer(); updateHandoffControls();
    try {
      const payload = addWorkflow({ project: handoff.project, memory: $("memory-toggle").checked });
      const prompt = $("history-prompt").value.trim(); if (prompt) payload.prompt = prompt;
      const result = await api(`/api/history/chats/${encodeURIComponent(handoff.selected)}/continue`, { method: "POST", body: JSON.stringify(payload), timeout: 60000 });
      const task = result.task || result; if (!task.id) throw new Error("The server did not return the created task.");
      mergeTask(task); $("history-dialog").close(); $("prompt").value = ""; resizePrompt();
      await selectTask(task.id); setInspector(!window.matchMedia("(max-width: 1000px)").matches); selectInspectorTab("activity");
      toast("History context transferred to CorePilot.");
    } catch (error) { $("history-error").textContent = displayError(error); $("history-error").hidden = false; }
    finally { handoff.busy = false; state.busy = false; updateComposer(); updateHandoffControls(); }
  }

  async function submitTask(event) {
    event.preventDefault();
    if ($("send-button").disabled) return;
    const prompt = $("prompt").value.trim(); const previous = currentTask();
    state.busy = true; clearError(); updateComposer();
    try {
      const payload = addWorkflow({ prompt, project: previous?.project_id || previous?.project || state.project, memory: $("memory-toggle").checked });
      if (previous?.status === "completed") payload.parent_id = previous.id;
      const resume = resumableStatuses.has(previous?.status);
      const result = await api(resume ? `/api/tasks/${encodeURIComponent(previous.id)}/resume` : "/api/tasks", { method: "POST", body: JSON.stringify(resume ? { prompt } : payload), timeout: 60000 });
      const task = result.task || result; mergeTask(task);
      $("prompt").value = ""; resizePrompt();
      await selectTask(task.id);
      setInspector(!window.matchMedia("(max-width: 1000px)").matches); selectInspectorTab("activity");
    } catch (error) { showError(error); }
    finally { state.busy = false; updateComposer(); }
  }

  async function taskAction(action) {
    const task = currentTask(); if (!task || state.busy) return;
    state.busy = true; clearError(); updateComposer();
    try {
      const result = await api(`/api/tasks/${encodeURIComponent(task.id)}/${action}`, { method: "POST", body: "{}", timeout: 60000 });
      mergeTask(result.task || result); await refreshTask(true); renderTaskList();
      toast(action === "cancel" ? "Stop requested. The checkpoint is saved after the current step." : "Task resumed with its saved context.");
    } catch (error) { showError(error); }
    finally { state.busy = false; updateComposer(); }
  }

  async function applyChanges() {
    const task = currentTask(); if (!task || state.busy || task.applied || activeStatuses.has(task.status)) return;
    state.busy = true; $("apply-changes").disabled = true; clearError();
    try {
      const result = await api(`/api/tasks/${encodeURIComponent(task.id)}/apply`, { method: "POST", body: "{}", timeout: 60000 });
      if (result.task?.id || result.id) mergeTask(result.task || result); else mergeTask({ ...task, applied: true });
      await refreshTask(true); await refreshDiff(); toast("Changes applied to the source repository.");
    } catch (error) { showError(error); $("apply-changes").disabled = false; }
    finally { state.busy = false; updateComposer(); }
  }

  function setInspector(open) { state.inspector = open; $("inspector").hidden = !open; $("inspector-toggle").setAttribute("aria-expanded", String(open)); if (open && state.inspectorTab === "diff") refreshDiff(); }
  function selectInspectorTab(tab) { state.inspectorTab = tab; for (const name of ["activity", "diff"]) { const active = name === tab; $(`${name}-tab`).classList.toggle("active", active); $(`${name}-tab`).setAttribute("aria-selected", String(active)); $(`${name}-panel`).hidden = !active; } if (tab === "diff") refreshDiff(); }
  function syncSidebarAccessibility() { $("sidebar").inert = window.matchMedia("(max-width: 760px)").matches && !$("sidebar").classList.contains("open"); }
  function closeSidebar() { $("sidebar").classList.remove("open"); $("sidebar-backdrop").classList.remove("open"); $("menu-toggle").setAttribute("aria-expanded", "false"); syncSidebarAccessibility(); }
  function resizePrompt() { const input = $("prompt"); input.style.height = "auto"; input.style.height = Math.min(180, Math.max(46, input.scrollHeight)) + "px"; }
  async function poll() {
    if (state.polling) return;
    state.polling = true;
    try {
      const now = Date.now();
      const calls = [];
      if (now - state.lastHealthRefresh > 15000 || !state.connected) calls.push(refreshHealth());
      if (now - state.lastTaskRefresh > 8000) calls.push(refreshTasks());
      if (state.selected) calls.push(refreshTask());
      await Promise.allSettled(calls);
      if (state.inspector && state.inspectorTab === "diff" && activeStatuses.has(currentTask()?.status)) await refreshDiff();
    } finally { state.polling = false; pollTimer = setTimeout(poll, document.hidden ? 5000 : activeStatuses.has(currentTask()?.status) ? 1200 : 3000); }
  }

  $("composer").addEventListener("submit", submitTask);
  $("prompt").addEventListener("input", () => { resizePrompt(); updateComposer(); clearError(); });
  $("prompt").addEventListener("keydown", (event) => { if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); if (!$("send-button").disabled) $("composer").requestSubmit(); } });
  $("new-task").addEventListener("click", newTask);
  $("import-history").addEventListener("click", openHistoryDialog);
  $("welcome-import").addEventListener("click", openHistoryDialog);
  $("close-history").addEventListener("click", () => $("history-dialog").close());
  $("history-dialog").addEventListener("cancel", (event) => { if (handoff.busy) event.preventDefault(); });
  $("history-project").addEventListener("change", (event) => { handoff.project = event.target.value; loadHistoryChats(); });
  $("history-provider").addEventListener("change", event => { handoff.provider = event.target.value; loadHistoryChats(); });
  $("refresh-history").addEventListener("click", () => loadHistoryChats(true));
  $("continue-history").addEventListener("click", continueHistoryChat);
  $("history-chat-list").addEventListener("keydown", (event) => {
    if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key) || handoff.busy || !handoff.chats.length) return;
    event.preventDefault();
    const current = handoff.chats.findIndex((chat) => chat.id === handoff.selected);
    const index = event.key === "Home" ? 0 : event.key === "End" ? handoff.chats.length - 1 : event.key === "ArrowDown" ? Math.min(handoff.chats.length - 1, current + 1) : Math.max(0, current - 1);
    handoff.selected = handoff.chats[index].id; renderHandoffList(); updateHandoffNote();
    $("history-chat-list").querySelectorAll(".history-chat-option")[index]?.focus();
  });
  $("project-select").addEventListener("change", (event) => setProject(event.target.value));
  $("workflow-select").addEventListener("change", () => { state.lastHealthRefresh = 0; renderModel(); refreshHealth(); });
  $("stop-button").addEventListener("click", () => taskAction("cancel"));
  $("resume-task").addEventListener("click", () => taskAction("resume"));
  $("apply-changes").addEventListener("click", applyChanges);
  $("inspector-toggle").addEventListener("click", () => setInspector(!state.inspector));
  $("inspector-close").addEventListener("click", () => setInspector(false));
  $("activity-tab").addEventListener("click", () => selectInspectorTab("activity"));
  $("diff-tab").addEventListener("click", () => selectInspectorTab("diff"));
  $("menu-toggle").addEventListener("click", () => { const open = !$("sidebar").classList.contains("open"); $("sidebar").classList.toggle("open", open); $("sidebar-backdrop").classList.toggle("open", open); $("menu-toggle").setAttribute("aria-expanded", String(open)); syncSidebarAccessibility(); });
  $("sidebar-backdrop").addEventListener("click", closeSidebar);
  $("model-button").addEventListener("click", () => { const open = $("model-popover").hidden; $("model-popover").hidden = !open; $("model-button").setAttribute("aria-expanded", String(open)); });
  document.addEventListener("click", (event) => { if (!event.target.closest(".model-menu-wrap")) { $("model-popover").hidden = true; $("model-button").setAttribute("aria-expanded", "false"); } });
  document.addEventListener("keydown", (event) => { if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k" && !$("history-dialog").open) { event.preventDefault(); newTask(); } if (event.key === "Escape" && !$("history-dialog").open) { closeSidebar(); $("model-popover").hidden = true; $("model-button").setAttribute("aria-expanded", "false"); if (state.inspector) setInspector(false); } });
  $("release-resources").addEventListener("click", async () => {
    if (!$("resource-idle-confirm").checked) { toast("Confirm the model server has stopped before clearing the hold."); return; }
    try { await api("/api/resources/release", {method: "POST", body: JSON.stringify({backend_confirmed_idle: true})}); $("resource-idle-confirm").checked = false; await refreshHealth(); toast("Resource hold cleared. Start the model server when you are ready."); }
    catch (error) { showError(error); }
  });
  $("refresh-status").addEventListener("click", async () => { await refreshHealth(); toast(state.model?.ready ? "Local model is ready." : "Connection status refreshed."); });
  $("reconnect").addEventListener("click", async () => { await Promise.allSettled([refreshHealth(), refreshTasks()]); });
  $("copy-diff").addEventListener("click", async () => { try { await navigator.clipboard.writeText(state.diff); toast("Diff copied."); } catch { toast("Clipboard access unavailable. Select the diff text manually."); } });
  $("copy-setup-prompt").addEventListener("click", async () => {
    try { await navigator.clipboard.writeText($("setup-agent-prompt").textContent); toast("Setup prompt copied. Paste it into your coding agent."); }
    catch { $("setup-prompt-details").open = true; toast("Clipboard access unavailable. Copy the setup prompt shown below."); }
  });
  for (const button of document.querySelectorAll(".suggestion")) button.addEventListener("click", () => { $("prompt").value = button.dataset.prompt; resizePrompt(); updateComposer(); $("prompt").focus(); });
  window.addEventListener("beforeunload", () => clearTimeout(pollTimer));
  window.addEventListener("resize", syncSidebarAccessibility);

  async function start() {
    syncSidebarAccessibility();
    try { state.project = localStorage.getItem("corepilot-project") || ""; } catch { /* Storage can be disabled by browser preferences. */ }
    await Promise.allSettled([refreshHealth(), refreshTasks()]);
    let saved; try { saved = localStorage.getItem("corepilot-task"); } catch { saved = null; }
    if (saved && state.tasks.has(saved)) await selectTask(saved);
    else updateComposer();
    poll();
  }
  start();
})();
