#include "compiler.h"
#include "cache.h"
#include "runtime.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>
#include <iostream>
#include <map>
#include <stdexcept>

namespace {
using Clock = std::chrono::steady_clock;
std::vector<std::vector<double>> readBuffers(const llvm::json::Object &request) {
  auto *inputs = request.getArray("buffers");
  if (!inputs || inputs->size() != 3) throw std::runtime_error("buffers must contain three arrays");
  std::vector<std::vector<double>> values;
  for (const auto &item : *inputs) {
    auto *array = item.getAsArray();
    if (!array) throw std::runtime_error("buffer is not an array");
    std::vector<double> buffer;
    for (const auto &value : *array) {
      auto number = value.getAsNumber();
      if (!number) throw std::runtime_error("buffer contains a non-number");
      buffer.push_back(*number);
    }
    values.push_back(std::move(buffer));
  }
  return values;
}
} // namespace

int main() {
  rlmlir::Compiler compiler;
  rlmlir::ExecutableCache cache;
  std::map<int64_t, std::shared_ptr<rlmlir::CompiledKernel>> handles;
  int64_t nextHandle = 1;
  std::string line;
  while (std::getline(std::cin, line)) {
    llvm::json::Object response;
    try {
      auto parsed = llvm::json::parse(line);
      if (!parsed) throw std::runtime_error(llvm::toString(parsed.takeError()));
      auto *request = parsed->getAsObject();
      if (!request) throw std::runtime_error("request must be a JSON object");
      auto command = request->getString("command");
      if (!command) throw std::runtime_error("command is required");
      if (line.size() > 64000000) throw std::runtime_error("request exceeds 64 MB");
      if (*command == "info") {
        response["llvm_version"] = "18.1.3";
        response["target"] = "x86-64-v3";
        response["schema_version"] = 1;
        response["cache_entries"] = static_cast<int64_t>(cache.size());
        response["active_handles"] = static_cast<int64_t>(handles.size());
      } else if (*command == "analyze" || *command == "compile") {
        auto source = request->getString("source");
        if (!source) throw std::runtime_error("source is required");
        if (source->size() > 1000000) throw std::runtime_error("source exceeds 1 MB");
        auto *actions = request->getArray("actions");
        if (request->get("actions") && !actions) throw std::runtime_error("actions must be an array");
        llvm::json::Array empty;
        if (*command == "analyze") {
          response = compiler.analyze(source->str(), actions ? *actions : empty);
        } else {
        if (handles.size() >= 128) throw std::runtime_error("active handle limit reached");
        std::string key = "llvm18.1.3|x86-64-v3|O3|strict-fp64|" + source->str() + "\n";
        llvm::raw_string_ostream stream(key);
        llvm::json::Array keyActions = actions ? *actions : empty;
        stream << llvm::json::Value(std::move(keyActions));
        auto kernel = cache.find(key);
        const bool hit = static_cast<bool>(kernel);
        if (!hit) {
          cache.reserveSlot();
          kernel = compiler.compile(source->str(), actions ? *actions : empty);
          cache.insert(key, kernel);
        }
        auto id = nextHandle++;
        handles[id] = kernel;
        response["handle"] = id;
        response["cache_hit"] = hit;
        response["lowering_ms"] = hit ? 0 : kernel->loweringMs;
        response["codegen_ms"] = hit ? 0 : kernel->codegenMs;
        response["lowered_ir"] = kernel->loweredIR;
        response["transformed_ir"] = kernel->transformedIR;
        response["llvm_ir"] = kernel->llvmIR;
        }
      } else if (*command == "invoke" || *command == "benchmark") {
        auto id = request->getInteger("handle");
        if (!id || !handles.count(*id)) throw std::runtime_error("unknown executable handle");
        rlmlir::Invocation invocation(handles.at(*id), readBuffers(*request));
        if (request->get("count") && !request->getInteger("count"))
          throw std::runtime_error("count must be an integer");
        int64_t count = request->getInteger("count").value_or(1);
        if (count < 1 || count > 10000000) throw std::runtime_error("count outside 1..10000000");
        // JSON decoding and buffer allocation are orchestration costs, outside execution.
        auto start = Clock::now();
        for (int64_t n = 0; n < count; ++n) invocation.run();
        response["batch_ms"] = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        response["count"] = count;
        llvm::json::Array output;
        for (double value : invocation.output()) output.push_back(value);
        response["output"] = std::move(output);
      } else if (*command == "release") {
        auto id = request->getInteger("handle");
        if (!id || !handles.erase(*id)) throw std::runtime_error("unknown executable handle");
      } else if (*command == "clear_cache") {
        cache.clearUnused();
        response["remaining_entries"] = static_cast<int64_t>(cache.size());
      } else if (*command == "dump_object") {
        auto id = request->getInteger("handle");
        auto path = request->getString("path");
        if (!id || !handles.count(*id) || !path || path->empty())
          throw std::runtime_error("dump_object requires a live handle and output path");
        handles.at(*id)->engine->dumpToObjectFile(*path);
      } else {
        throw std::runtime_error("unknown command");
      }
      response["ok"] = true;
    } catch (const std::exception &error) {
      response["ok"] = false;
      response["error"] = error.what();
    }
    llvm::outs() << llvm::json::Value(std::move(response)) << '\n';
    llvm::outs().flush();
  }
}
