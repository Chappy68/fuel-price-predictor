const MODEL_PATHS = ["./model.json", "../docs/model.json"];
const LABEL_OVERRIDES = {
  brent_crude_usd: "Brent crude price (USD per barrel)",
  tax_percentage: "Fuel tax percentage",
  date_year: "Year",
  date_month: "Month",
  date_weekofyear: "ISO week of year",
  country: "Country",
  region: "Region",
  income_level: "Income level",
  subsidy_level: "Subsidy level",
};

let modelData = null;

const formatPrice = (value) =>
  new Intl.NumberFormat("en-US", {
    minimumFractionDigits: 3,
    maximumFractionDigits: 3,
  }).format(value);

const formatMetric = (value) =>
  new Intl.NumberFormat("en-US", {
    minimumFractionDigits: 3,
    maximumFractionDigits: 3,
  }).format(value);

const prettifyName = (name) => LABEL_OVERRIDES[name] || name.replaceAll("_", " ");

async function loadModel() {
  for (const path of MODEL_PATHS) {
    try {
      const response = await fetch(path);
      if (response.ok) {
        return await response.json();
      }
    } catch (error) {
      continue;
    }
  }
  throw new Error("Unable to load model.json. Serve the app through a local server or GitHub Pages.");
}

function setText(id, value) {
  const element = document.getElementById(id);
  if (element) {
    element.textContent = value;
  }
}

function createNumericInput(field) {
  const wrapper = document.createElement("div");
  wrapper.className = "input-card";

  const label = document.createElement("label");
  label.htmlFor = field.name;
  label.innerHTML = `<span>${prettifyName(field.name)}</span><output id="${field.name}-value">${field.default}</output>`;

  const input = document.createElement("input");
  input.type = "range";
  input.id = field.name;
  input.name = field.name;
  input.min = field.min;
  input.max = field.max;
  input.step = field.step;
  input.value = field.default;
  input.dataset.default = field.default;

  input.addEventListener("input", () => {
    document.getElementById(`${field.name}-value`).textContent = Number(input.value).toFixed(
      field.step >= 1 ? 0 : 1
    );
    renderPrediction();
  });

  const caption = document.createElement("p");
  caption.className = "input-caption";
  caption.textContent = `Range: ${field.min} to ${field.max}`;

  wrapper.append(label, input, caption);
  return wrapper;
}

function createCategoricalInput(field) {
  const wrapper = document.createElement("div");
  wrapper.className = "input-card";

  const label = document.createElement("label");
  label.htmlFor = field.name;
  label.innerHTML = `<span>${prettifyName(field.name)}</span><output>${field.default}</output>`;

  const select = document.createElement("select");
  select.id = field.name;
  select.name = field.name;
  select.dataset.default = field.default;

  for (const optionValue of field.options) {
    const option = document.createElement("option");
    option.value = optionValue;
    option.textContent = optionValue;
    if (optionValue === field.default) {
      option.selected = true;
    }
    select.append(option);
  }

  select.addEventListener("change", () => {
    label.querySelector("output").textContent = select.value;
    renderPrediction();
  });

  wrapper.append(label, select);
  return wrapper;
}

function populateInputs(model) {
  const numericRoot = document.getElementById("numeric-inputs");
  const categoricalRoot = document.getElementById("categorical-inputs");
  numericRoot.innerHTML = "";
  categoricalRoot.innerHTML = "";

  model.inputSchema.numeric.forEach((field) => numericRoot.append(createNumericInput(field)));
  model.inputSchema.categorical.forEach((field) => categoricalRoot.append(createCategoricalInput(field)));
}

function getFormState() {
  const values = {};
  document.querySelectorAll("#prediction-form input, #prediction-form select").forEach((element) => {
    values[element.name] = element.tagName === "SELECT" ? element.value : Number(element.value);
  });
  return values;
}

function getNumericContribution(feature, inputValues, coefficients) {
  const rawValue = Number.isFinite(inputValues[feature.name]) ? inputValues[feature.name] : feature.imputer;
  const standardized = (rawValue - feature.mean) / feature.scale;
  const key = `num__${feature.name}`;
  return {
    label: prettifyName(feature.name),
    value: standardized * (coefficients[key] || 0),
    detail: rawValue,
  };
}

function getCategoricalContribution(feature, inputValues, coefficients) {
  const selectedValue = inputValues[feature.name] || feature.mode || feature.baseCategory;
  const key = `cat__${feature.name}_${selectedValue}`;
  return {
    label: prettifyName(feature.name),
    value: selectedValue === feature.baseCategory ? 0 : coefficients[key] || 0,
    detail: selectedValue,
  };
}

function computePrediction(model, inputValues) {
  const contributions = [
    { label: "Intercept", value: model.intercept, detail: "Model baseline" },
    ...model.numericFeatures.map((feature) =>
      getNumericContribution(feature, inputValues, model.coefficients)
    ),
    ...model.categoricalFeatures.map((feature) =>
      getCategoricalContribution(feature, inputValues, model.coefficients)
    ),
  ];

  const prediction = contributions.reduce((sum, item) => sum + item.value, 0);
  return { prediction, contributions };
}

function renderContributionList(contributions) {
  const root = document.getElementById("contribution-list");
  root.innerHTML = "";

  const nonIntercept = contributions.filter((item) => item.label !== "Intercept");
  const maxAbs = Math.max(...nonIntercept.map((item) => Math.abs(item.value)), 0.001);

  nonIntercept
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .forEach((item) => {
      const row = document.createElement("div");
      row.className = "contribution-item";

      const name = document.createElement("div");
      name.className = "contribution-name";
      name.textContent = `${item.label}: ${item.detail}`;

      const bar = document.createElement("div");
      bar.className = "contribution-bar";

      const fill = document.createElement("div");
      fill.className = "contribution-fill";
      fill.style.width = `${(Math.abs(item.value) / maxAbs) * 100}%`;
      fill.style.background = item.value >= 0 ? "var(--positive)" : "var(--negative)";
      bar.append(fill);

      const value = document.createElement("div");
      value.className = "contribution-value";
      value.textContent = `${item.value >= 0 ? "+" : ""}${formatPrice(item.value)}`;

      row.append(name, bar, value);
      root.append(row);
    });
}

function renderMetrics(model) {
  const metricRoot = document.getElementById("metric-cards");
  metricRoot.innerHTML = "";

  const cards = [
    { label: "MAE", value: formatMetric(model.metrics.mae) },
    { label: "RMSE", value: formatMetric(model.metrics.rmse) },
    { label: "R^2", value: formatMetric(model.metrics.r2) },
  ];

  cards.forEach((metric) => {
    const card = document.createElement("div");
    card.className = "metric-card";
    card.innerHTML = `<span>${metric.label}</span><strong>${metric.value}</strong>`;
    metricRoot.append(card);
  });
}

function renderDatasetInfo(model) {
  setText("dataset-rows", model.dataset.rows.toLocaleString("en-US"));
  setText("train-rows", model.dataset.trainingRows.toLocaleString("en-US"));
  setText("test-rows", model.dataset.testRows.toLocaleString("en-US"));
  setText("data-file", model.dataset.fileName);
  setText("target-name", `${prettifyName(model.targetColumn)} (USD/L)`);
}

function renderPrediction() {
  if (!modelData) {
    return;
  }

  const state = getFormState();
  const result = computePrediction(modelData, state);
  const nonIntercept = result.contributions.filter((item) => item.label !== "Intercept");
  const topPositive = [...nonIntercept].sort((a, b) => b.value - a.value)[0];
  const topNegative = [...nonIntercept].sort((a, b) => a.value - b.value)[0];

  setText("prediction-value", `${formatPrice(result.prediction)} USD/L`);
  setText("prediction-subtitle", `Target: ${prettifyName(modelData.targetColumn)}`);
  setText("intercept-value", `${formatPrice(modelData.intercept)} USD/L`);
  setText("top-positive", topPositive ? `${topPositive.label} (${formatPrice(topPositive.value)})` : "-");
  setText("top-negative", topNegative ? `${topNegative.label} (${formatPrice(topNegative.value)})` : "-");

  renderContributionList(result.contributions);
}

function resetForm() {
  document.querySelectorAll("#prediction-form input, #prediction-form select").forEach((element) => {
    if (element.dataset.default !== undefined) {
      element.value = element.dataset.default;
      if (element.tagName === "INPUT") {
        document.getElementById(`${element.name}-value`).textContent = Number(element.value).toFixed(
          Number(element.step) >= 1 ? 0 : 1
        );
      } else {
        const card = element.closest(".input-card");
        const output = card?.querySelector("output");
        if (output) {
          output.textContent = element.value;
        }
      }
    }
  });
  renderPrediction();
}

async function init() {
  try {
    modelData = await loadModel();
    populateInputs(modelData);
    renderMetrics(modelData);
    renderDatasetInfo(modelData);
    renderPrediction();
    document.getElementById("reset-button").addEventListener("click", resetForm);
  } catch (error) {
    setText("prediction-value", "Model loading failed");
    setText("prediction-subtitle", error.message);
  }
}

if (typeof document !== "undefined") {
  init();
}
