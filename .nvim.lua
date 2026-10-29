-- Project-local Neovim config: attach the Axiom LSP (`axiom --lsp`) to code
-- buffers when an Axiom binary is available.
--
-- Resolution order (first hit wins):
--   1. $AXIOM_BIN, if set
--   2. `axiom` on PATH (installed via install.sh / cargo install / pip)
--   3. a release build in this repo: axiom_engine_rs/target/release/axiom[_engine]
-- Nothing is hardcoded to a particular machine or drive letter, so the same
-- file works for every clone. Neovim only loads this with `exrc` enabled.

local uv = vim.uv or vim.loop
local repo = vim.fn.fnamemodify(vim.fn.expand("<sfile>:p"), ":h")
local exe_suffix = (uv.os_uname().sysname == "Windows_NT") and ".exe" or ""

local function first_executable(candidates)
  for _, cmd in ipairs(candidates) do
    if cmd and cmd ~= "" and vim.fn.executable(cmd) == 1 then
      return cmd
    end
  end
  return nil
end

local axiom_cmd = first_executable({
  os.getenv("AXIOM_BIN"),
  "axiom",
  repo .. "/axiom_engine_rs/target/release/axiom" .. exe_suffix,
  repo .. "/axiom_engine_rs/target/release/axiom_engine" .. exe_suffix,
})

if axiom_cmd then
  local function axiom_root(file)
    local markers = vim.fs.find({ "Cargo.toml", ".git" }, { path = file, upward = true })
    return markers[1] and vim.fs.dirname(markers[1]) or uv.cwd()
  end

  vim.api.nvim_create_autocmd("FileType", {
    pattern = { "rust", "javascript", "typescript", "python" },
    callback = function(args)
      local root_dir = axiom_root(args.file)
      local clients = vim.lsp.get_clients({ name = "axiom_engine" })
      for _, client in ipairs(clients) do
        if client.config.root_dir == root_dir then
          vim.lsp.buf_attach_client(args.buf, client.id)
          return
        end
      end

      vim.lsp.start({
        name = "axiom_engine",
        cmd = { axiom_cmd, "--lsp" },
        root_dir = root_dir,
        settings = {},
      }, { bufnr = args.buf })
    end,
  })
end
