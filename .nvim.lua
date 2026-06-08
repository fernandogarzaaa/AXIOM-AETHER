local axiom_cmd = "C:/Users/garza/AXIOM-AETHER/target/rt_serve_release/release/axiom_engine.exe"

if vim.fn.executable(axiom_cmd) == 1 then
  local function axiom_root(file)
    local markers = vim.fs.find({ "Cargo.toml", ".git" }, { path = file, upward = true })
    return markers[1] and vim.fs.dirname(markers[1]) or vim.loop.cwd()
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

      local client_id = vim.lsp.start_client({
        name = "axiom_engine",
        cmd = { axiom_cmd, "--lsp" },
        root_dir = root_dir,
        settings = {},
      })
      if client_id then
        vim.lsp.buf_attach_client(args.buf, client_id)
      end
    end,
  })
end
